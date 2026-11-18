let session = null;
let labelMap = {};
let refEmbeddings = {};
let refSequences = {};
let topicsData = {};
let allWords = [];
let currentTopic = "Tất cả";
let currentLearnWord = "";
let practiceTopic = "Tất cả";
let quizTopic = "Tất cả";

let isRecording = false;
let recordedFrames = [];
let animTimers = {};

// Biến cho Quiz & Trò chơi
let quizMode = "mcq"; // 'mcq' | 'match' | 'speed'
let currentQuizWord = "";
let quizPoints = 0;
let speedInterval = null;
let speedTimeLeft = 60;
let speedCombo = 1;
let selectedMatchVid = null;
let selectedMatchWord = null;
let matchedCount = 0;

const MAX_LEN = 64;
const videoElement = document.getElementById("webcam");
const wordSelect = document.getElementById("wordSelect");
const recordBtn = document.getElementById("recordBtn");
const statusEl = document.getElementById("status");

function switchTab(tabId, btnEl) {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(tabId).classList.add("active");
  btnEl.classList.add("active");

  if (tabId === "quiz") {
    renderLeaderboard();
    if (quizMode === "match") initMatchGame();
    else if (!currentQuizWord) nextQuizQuestion();
  }
}

// ================= 1. KHỞI TẠO DỮ LIỆU =================
async function init() {
  try {
    statusEl.textContent = "1/4 Đang tải danh sách từ vựng & chủ đề...";
    const labelsRes = await fetch("./models/label_map.json");
    if (!labelsRes.ok) throw new Error("Không tìm thấy label_map.json");
    labelMap = await labelsRes.json();
    allWords = Object.keys(labelMap).sort();

    const topicsRes = await fetch("./models/topics.json");
    if (topicsRes.ok) {
      topicsData = await topicsRes.json();
    } else {
      topicsData = { "Tất cả từ vựng": allWords };
    }

    statusEl.textContent = "2/4 Đang tải vector chuẩn...";
    const refRes = await fetch("./models/reference_embeddings.json");
    refEmbeddings = await refRes.json();

    statusEl.textContent = "3/4 Đang tải dữ liệu khung xương dự phòng...";
    const seqRes = await fetch("./models/reference_sequences.json");
    refSequences = await seqRes.json();

    setupLearnModule();
    setupPracticeAndQuizTopics();

    statusEl.textContent = "4/4 Đang nạp mô hình AI ONNX...";
    session = await ort.InferenceSession.create("./models/vsl_scorer.onnx");

    statusEl.textContent = "Sẵn sàng thực hành!";
    statusEl.style.color = "#16a34a";
    recordBtn.textContent = "Bắt đầu quay (3 giây)";
    recordBtn.disabled = false;

    startCamera();
  } catch (err) {
    statusEl.style.color = "#dc2626";
    statusEl.textContent = "LỖI KHỞI TẠO: " + err.message;
  }
}

// ================= 2. MODULE HỌC & LỌC CHỦ ĐỀ CHO CẢ 3 TAB =================
function setupLearnModule() {
  const pillBox = document.getElementById("topicPills");
  pillBox.innerHTML = "";

  const topicNames = ["Tất cả", ...Object.keys(topicsData)];
  topicNames.forEach(tName => {
    const btn = document.createElement("button");
    btn.className = "topic-pill" + (tName === "Tất cả" ? " active" : "");
    btn.textContent = tName;
    btn.onclick = () => {
      document.querySelectorAll(".topic-pill").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      currentTopic = tName;
      filterLearnWords();
    };
    pillBox.appendChild(btn);
  });

  filterLearnWords();
  if (allWords.length > 0) selectLearnWord(allWords[0]);
}

function filterLearnWords() {
  const keyword = document.getElementById("learnSearchInput").value.trim().toLowerCase();
  const baseList = currentTopic === "Tất cả" ? allWords : (topicsData[currentTopic] || []);
  const filtered = baseList.filter(w => w.toLowerCase().includes(keyword));

  document.getElementById("learnWordCount").textContent = `Hiển thị ${filtered.length} từ vựng`;
  const grid = document.getElementById("learnWordGrid");
  grid.innerHTML = "";

  filtered.forEach(word => {
    const card = document.createElement("div");
    card.className = "word-card" + (word === currentLearnWord ? " selected" : "");
    card.textContent = word;
    card.onclick = () => selectLearnWord(word);
    grid.appendChild(card);
  });
}

function selectLearnWord(word) {
  currentLearnWord = word;
  document.getElementById("learnCurrentTitle").textContent = `Từ đang học: "${word}"`;
  filterLearnWords();
  showDemoMedia("learn", word);
}

function setupPracticeAndQuizTopics() {
  const topicNames = ["Tất cả", ...Object.keys(topicsData)];

  // Nạp chủ đề cho Tab Thực hành
  const pTopicSel = document.getElementById("practiceTopicSelect");
  pTopicSel.innerHTML = "";
  topicNames.forEach(t => {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    pTopicSel.appendChild(opt);
  });
  onPracticeTopicChange();

  // Nạp chủ đề cho Tab Quiz
  const qTopicSel = document.getElementById("quizTopicSelect");
  qTopicSel.innerHTML = "";
  topicNames.forEach(t => {
    const opt = document.createElement("option");
    opt.value = t;
    const count = t === "Tất cả" ? allWords.length : topicsData[t].length;
    opt.textContent = `${t} (${count} từ)`;
    qTopicSel.appendChild(opt);
  });
}

function onPracticeTopicChange() {
  practiceTopic = document.getElementById("practiceTopicSelect").value;
  const list = practiceTopic === "Tất cả" ? allWords : (topicsData[practiceTopic] || []);

  wordSelect.innerHTML = "";
  list.forEach(w => {
    const opt = document.createElement("option");
    opt.value = w;
    opt.textContent = w;
    wordSelect.appendChild(opt);
  });

  if (list.length > 0) {
    showDemoMedia("practice", list[0]);
  }
}

function getQuizWordPool() {
  if (quizTopic === "Tất cả") return Object.keys(refSequences);
  const topicWords = (topicsData[quizTopic] || []).filter(w => refSequences[w]);
  return topicWords.length >= 4 ? topicWords : Object.keys(refSequences);
}

function onQuizTopicChange() {
  quizTopic = document.getElementById("quizTopicSelect").value;
  if (quizMode === "match") {
    initMatchGame();
  } else {
    nextQuizQuestion();
  }
}

function goToPracticeCurrentWord() {
  if (!currentLearnWord) return;
  document.getElementById("practiceTopicSelect").value = "Tất cả";
  onPracticeTopicChange();
  wordSelect.value = currentLearnWord;
  showDemoMedia("practice", currentLearnWord);
  const practiceTabBtn = document.querySelectorAll(".tab-btn")[1];
  switchTab("practice", practiceTabBtn);
}

// ================= 3. HIỂN THỊ VIDEO MẪU / KHUNG XƯƠNG DỰ PHÒNG =================
function getVideoUrl(word) {
  const safeWord = word.replace(/[\/\\]/g, "_").replace(/\?/g, "");
  return `./videos/${encodeURIComponent(safeWord)}.mp4`;
}

async function showDemoMedia(mode, word) {
  const mapIds = {
    learn: { v: "learnVideo", c: "learnCanvas", b: "learnBadge" },
    practice: { v: "refVideo", c: "refCanvas", b: "practiceBadge" },
    quiz: { v: "quizVideo", c: "quizCanvas", b: "quizBadge" }
  };
  const { v: videoId, c: canvasId, b: badgeId } = mapIds[mode];
  const videoEl = document.getElementById(videoId);
  const canvasEl = document.getElementById(canvasId);
  const badgeEl = document.getElementById(badgeId);
  const videoUrl = getVideoUrl(word);

  let fileExists = false;
  try {
    const res = await fetch(videoUrl, { method: "HEAD" });
    fileExists = res.ok;
  } catch (e) {}

  if (fileExists) {
    if (animTimers[canvasId]) clearInterval(animTimers[canvasId]);
    canvasEl.style.display = "none";
    videoEl.style.display = "block";
    videoEl.src = videoUrl;
    videoEl.load();
    videoEl.play().catch(() => {});
    badgeEl.textContent = "Video Người mẫu thật";
    return;
  }

  videoEl.pause();
  videoEl.style.display = "none";
  canvasEl.style.display = "block";
  badgeEl.textContent = "Mô phỏng Khung xương AI";
  playFallbackSkeleton(canvasId, word);
}

function playFallbackSkeleton(canvasId, word) {
  if (animTimers[canvasId]) clearInterval(animTimers[canvasId]);
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const seq = refSequences[word];
  if (!seq) return;

  let frameIdx = 0;
  animTimers[canvasId] = setInterval(() => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const kp = seq[frameIdx];

    const connections = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 3;
    connections.forEach(([a, b]) => {
      ctx.beginPath();
      ctx.moveTo(kp[a][0] * 110 + canvas.width / 2, kp[a][1] * 110 + canvas.height * 0.45);
      ctx.lineTo(kp[b][0] * 110 + canvas.width / 2, kp[b][1] * 110 + canvas.height * 0.45);
      ctx.stroke();
    });

    for (let i = 34; i < 55; i++) {
      ctx.fillStyle = "#4ade80";
      ctx.beginPath();
      ctx.arc(kp[15][0] * 110 + canvas.width / 2 + kp[i][0] * 40, kp[15][1] * 110 + canvas.height * 0.45 + kp[i][1] * 40, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    for (let i = 55; i < 76; i++) {
      ctx.fillStyle = "#f472b6";
      ctx.beginPath();
      ctx.arc(kp[16][0] * 110 + canvas.width / 2 + kp[i][0] * 40, kp[16][1] * 110 + canvas.height * 0.45 + kp[i][1] * 40, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    frameIdx = (frameIdx + 1) % MAX_LEN;
  }, 45);
}

// ================= 4. CHẤM ĐIỂM & SỬA LỖI AI =================
function generateCorrectionFeedback(userSeq64, refSeq64, rawFrameCount) {
  const tips = [];
  let poseErr = 0, leftHandErr = 0, rightHandErr = 0;

  for (let t = 0; t < MAX_LEN; t++) {
    [13, 14, 15, 16].forEach(idx => {
      poseErr += Math.hypot(userSeq64[t][idx][0] - refSeq64[t][idx][0], userSeq64[t][idx][1] - refSeq64[t][idx][1]);
    });
    for (let i = 34; i < 55; i++) {
      leftHandErr += Math.hypot(userSeq64[t][i][0] - refSeq64[t][i][0], userSeq64[t][i][1] - refSeq64[t][i][1]);
    }
    for (let i = 55; i < 76; i++) {
      rightHandErr += Math.hypot(userSeq64[t][i][0] - refSeq64[t][i][0], userSeq64[t][i][1] - refSeq64[t][i][1]);
    }
  }

  poseErr /= (MAX_LEN * 4);
  leftHandErr /= (MAX_LEN * 21);
  rightHandErr /= (MAX_LEN * 21);

  if (poseErr > 0.35) tips.push("<b>Vị trí cánh tay:</b> Biên độ đưa tay chưa khớp video mẫu. Hãy chú ý độ cao và khoảng cách tay so với vai/ngực.");
  if (leftHandErr > 0.38) tips.push("<b>Bàn tay trái:</b> Hình dáng gập/duỗi các ngón tay trái chưa chuẩn. Hãy mở rộng bàn tay rõ ràng trước camera.");
  if (rightHandErr > 0.38) tips.push("<b>Bàn tay phải:</b> Khẩu hình ngón tay phải còn lệch so với video mẫu. Chú ý hướng lòng bàn tay và độ khép của các ngón.");
  if (rawFrameCount < 22) tips.push("<b>Tốc độ thực hiện:</b> Bạn làm động tác hơi nhanh, hãy giữ tư thế kết thúc thêm 0.5 giây để AI nhận diện trọn vẹn.");
  if (tips.length === 0) tips.push("<b>Tuyệt vời:</b> Vị trí tay, khẩu hình ngón tay và nhịp độ của bạn rất khớp với video mẫu!");
  return tips;
}

function extract76Keypoints(results) {
  const kp = [];
  const pose = results.poseLandmarks || Array(33).fill({ x: 0, y: 0, z: 0 });
  const lShoulder = pose[11] || { x: 0.5, y: 0.5, z: 0 };
  const rShoulder = pose[12] || { x: 0.5, y: 0.5, z: 0 };
  const mid = { x: (lShoulder.x + rShoulder.x) / 2, y: (lShoulder.y + rShoulder.y) / 2, z: (lShoulder.z + rShoulder.z) / 2 };
  const dist = Math.hypot(lShoulder.x - rShoulder.x, lShoulder.y - rShoulder.y) || 1.0;

  for (let i = 0; i < 33; i++) {
    kp.push([(pose[i].x - mid.x) / dist, (pose[i].y - mid.y) / dist, (pose[i].z - mid.z) / dist]);
  }
  kp.push([0, 0, 0]);

  const normHand = (hl) => {
    if (!hl || hl.length === 0) return Array(21).fill([0, 0, 0]);
    const xs = hl.map(p => p.x), ys = hl.map(p => p.y);
    const minX = Math.min(...xs), spanX = (Math.max(...xs) - minX) || 1e-6;
    const minY = Math.min(...ys), spanY = (Math.max(...ys) - minY) || 1e-6;
    const wz = hl[0].z;
    return hl.map(p => [(p.x - minX) / spanX - 0.5, (p.y - minY) / spanY - 0.5, p.z - wz]);
  };
  kp.push(...normHand(results.leftHandLandmarks));
  kp.push(...normHand(results.rightHandLandmarks));
  return kp;
}

function preprocessSequence(frames) {
  const T = frames.length;
  let sum = 0, count = T * 76 * 3;
  frames.forEach(f => f.forEach(pt => pt.forEach(v => sum += v)));
  const mean = sum / count;

  let sqSum = 0;
  frames.forEach(f => f.forEach(pt => pt.forEach(v => sqSum += (v - mean) ** 2)));
  const std = Math.sqrt(sqSum / count) + 1e-6;

  const norm3D = frames.map(f => f.map(pt => pt.map(v => (v - mean) / std)));
  const seq64 = [];
  const flat = new Float32Array(MAX_LEN * 228);

  for (let i = 0; i < MAX_LEN; i++) {
    const srcIdx = T >= MAX_LEN ? Math.floor(i * (T - 1) / (MAX_LEN - 1)) : i;
    const frame = srcIdx < T ? norm3D[srcIdx] : Array(76).fill([0, 0, 0]);
    seq64.push(frame);
    flat.set(frame.flat(), i * 228);
  }
  return { flat, seq64 };
}

async function evaluateSign() {
  if (recordedFrames.length < 10) {
    statusEl.textContent = "Không thu được khung hình nào từ camera! Hãy kiểm tra quyền Webcam.";
    return;
  }
  const { flat, seq64 } = preprocessSequence(recordedFrames);
  const tensor = new ort.Tensor("float32", flat, [1, MAX_LEN, 228]);
  const results = await session.run({ keypoints: tensor });

  const logits = results.logits.data;
  const userEmb = results.embedding.data;
  const targetWord = wordSelect.value;
  const targetIdx = labelMap[targetWord];

  const maxL = Math.max(...logits);
  const exps = Array.from(logits).map(v => Math.exp(v - maxL));
  const targetProb = exps[targetIdx] / exps.reduce((a, b) => a + b, 0);

  const refEmb = refEmbeddings[targetWord];
  let sim = 0;
  for (let i = 0; i < userEmb.length; i++) sim += userEmb[i] * refEmb[i];
  sim = Math.max(0, sim);

  const score = Math.round((0.4 * targetProb + 0.6 * sim) * 100);
  document.getElementById("scoreDisplay").textContent = `${score} / 100`;
  statusEl.textContent = score >= 75 ? "Đạt yêu cầu!" : "Cần chỉnh sửa thêm:";

  const tips = generateCorrectionFeedback(seq64, refSequences[targetWord], recordedFrames.length);
  document.getElementById("correctionList").innerHTML = tips.map(t => `<li>${t}</li>`).join("");
}

// ================= 5. QUIZ ĐA CHẾ ĐỘ & BẢNG XẾP HẠNG =================
function setQuizMode(mode) {
  quizMode = mode;
  clearInterval(speedInterval);
  quizPoints = 0;
  document.getElementById("quizScore").textContent = "0";

  document.getElementById("btnModeMcq").classList.toggle("active", mode === "mcq");
  document.getElementById("btnModeMatch").classList.toggle("active", mode === "match");
  document.getElementById("btnModeSpeed").classList.toggle("active", mode === "speed");

  document.getElementById("mcqContainer").style.display = (mode === "mcq" || mode === "speed") ? "block" : "none";
  document.getElementById("matchContainer").style.display = mode === "match" ? "block" : "none";
  document.getElementById("speedTimerBox").style.display = mode === "speed" ? "inline" : "none";
  document.getElementById("btnNextMcq").style.display = mode === "mcq" ? "inline-block" : "none";

  if (mode === "mcq") {
    nextQuizQuestion();
  } else if (mode === "match") {
    initMatchGame();
  } else if (mode === "speed") {
    startSpeedGame();
  }
}

// Chế độ 1 & 3: Trắc nghiệm & Thử thách 60s
function nextQuizQuestion() {
  const words = getQuizWordPool();
  currentQuizWord = words[Math.floor(Math.random() * words.length)];
  const choices = new Set([currentQuizWord]);
  while (choices.size < 4) {
    choices.add(words[Math.floor(Math.random() * words.length)]);
  }

  const shuffled = Array.from(choices).sort(() => Math.random() - 0.5);
  const optBox = document.getElementById("quizOptions");
  optBox.innerHTML = "";
  document.getElementById("quizFeedback").textContent = "";

  shuffled.forEach(word => {
    const btn = document.createElement("button");
    btn.className = "quiz-opt";
    btn.textContent = word;
    btn.onclick = () => {
      document.querySelectorAll(".quiz-opt").forEach(b => b.disabled = true);
      if (word === currentQuizWord) {
        btn.classList.add("correct");
        const pts = quizMode === "speed" ? (10 * speedCombo) : 10;
        quizPoints += pts;
        if (quizMode === "speed") speedCombo++;
        document.getElementById("quizScore").textContent = quizPoints;
        document.getElementById("quizFeedback").textContent = `Chính xác! +${pts} điểm` + (quizMode === "speed" ? ` (Combo x${speedCombo - 1})` : "");
      } else {
        btn.classList.add("wrong");
        if (quizMode === "speed") speedCombo = 1;
        document.getElementById("quizFeedback").textContent = `Sai rồi! Đáp án đúng là: "${currentQuizWord}"`;
      }

      if (quizMode === "speed" && speedTimeLeft > 0) {
        setTimeout(() => nextQuizQuestion(), 650);
      }
    };
    optBox.appendChild(btn);
  });

  showDemoMedia("quiz", currentQuizWord);
}

function replayQuizAnim() {
  if (currentQuizWord) showDemoMedia("quiz", currentQuizWord);
}

function startSpeedGame() {
  speedTimeLeft = 60;
  speedCombo = 1;
  document.getElementById("speedTimer").textContent = speedTimeLeft;
  nextQuizQuestion();

  speedInterval = setInterval(() => {
    speedTimeLeft--;
    document.getElementById("speedTimer").textContent = speedTimeLeft;
    if (speedTimeLeft <= 0) {
      clearInterval(speedInterval);
      document.querySelectorAll(".quiz-opt").forEach(b => b.disabled = true);
      document.getElementById("quizFeedback").textContent = `⏰ HẾT GIỜ! Tổng điểm của bạn: ${quizPoints}. Hãy nhập tên để lưu vào Bảng Xếp Hạng!`;
    }
  }, 1000);
}

// Chế độ 2: Nối từ với Video
function initMatchGame() {
  selectedMatchVid = null;
  selectedMatchWord = null;
  matchedCount = 0;

  const words = getQuizWordPool();
  const picked = new Set();
  while (picked.size < 4) {
    picked.add(words[Math.floor(Math.random() * words.length)]);
  }
  const pairWords = Array.from(picked);
  const shuffledWords = [...pairWords].sort(() => Math.random() - 0.5);

  const vidCol = document.getElementById("matchVideoCol");
  const wordCol = document.getElementById("matchWordCol");
  vidCol.innerHTML = "";
  wordCol.innerHTML = "";

  pairWords.forEach(w => {
    const card = document.createElement("div");
    card.className = "match-vid-card";
    card.dataset.word = w;
    card.innerHTML = `<video src="${getVideoUrl(w)}" autoplay loop muted playsinline></video>`;
    card.onclick = () => {
      document.querySelectorAll(".match-vid-card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      selectedMatchVid = card;
      checkMatchPair();
    };
    vidCol.appendChild(card);
  });

  shuffledWords.forEach(w => {
    const btn = document.createElement("button");
    btn.className = "match-word-btn";
    btn.dataset.word = w;
    btn.textContent = w;
    btn.onclick = () => {
      document.querySelectorAll(".match-word-btn").forEach(b => b.classList.remove("selected"));
      btn.classList.add("selected");
      selectedMatchWord = btn;
      checkMatchPair();
    };
    wordCol.appendChild(btn);
  });
}

function checkMatchPair() {
  if (!selectedMatchVid || !selectedMatchWord) return;
  if (selectedMatchVid.dataset.word === selectedMatchWord.dataset.word) {
    selectedMatchVid.classList.remove("selected");
    selectedMatchWord.classList.remove("selected");
    selectedMatchVid.classList.add("matched");
    selectedMatchWord.classList.add("matched");
    quizPoints += 15;
    matchedCount++;
    document.getElementById("quizScore").textContent = quizPoints;
    selectedMatchVid = null;
    selectedMatchWord = null;

    if (matchedCount === 4) {
      setTimeout(() => initMatchGame(), 800);
    }
  } else {
    setTimeout(() => {
      if (selectedMatchVid) selectedMatchVid.classList.remove("selected");
      if (selectedMatchWord) selectedMatchWord.classList.remove("selected");
      selectedMatchVid = null;
      selectedMatchWord = null;
    }, 350);
  }
}

// Bảng xếp hạng (Lưu trữ cục bộ trên trình duyệt)
function saveScoreToLeaderboard() {
  const nameInput = document.getElementById("playerNameInput");
  const playerName = nameInput.value.trim() || "Học viên ẩn danh";
  if (quizPoints <= 0) {
    alert("Bạn cần đạt ít nhất 10 điểm để ghi danh lên bảng xếp hạng!");
    return;
  }

  const modeLabels = { mcq: "Trắc nghiệm", match: "Nối Video", speed: "Thử thách 60s" };
  const board = JSON.parse(localStorage.getItem("vsl_leaderboard") || "[]");
  board.push({ name: playerName, mode: modeLabels[quizMode], score: quizPoints });
  board.sort((a, b) => b.score - a.score);
  localStorage.setItem("vsl_leaderboard", JSON.stringify(board.slice(0, 10)));

  nameInput.value = "";
  renderLeaderboard();
}

function renderLeaderboard() {
  const board = JSON.parse(localStorage.getItem("vsl_leaderboard") || "[]");
  const tbody = document.getElementById("leaderboardBody");
  if (board.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; color:#64748b;">Chưa có kỷ lục nào. Hãy chơi và lưu điểm!</td></tr>`;
    return;
  }

  tbody.innerHTML = board.map((item, idx) => {
    const rankClass = idx === 0 ? "rank-1" : idx === 1 ? "rank-2" : idx === 2 ? "rank-3" : "";
    const medal = idx === 0 ? "🥇" : idx === 1 ? "🥈" : idx === 2 ? "🥉" : `#${idx + 1}`;
    return `<tr class="${rankClass}">
      <td>${medal}</td>
      <td>${item.name}</td>
      <td>${item.mode}</td>
      <td>${item.score}</td>
    </tr>`;
  }).join("");
}

// ================= 6. CAMERA & MEDIAPIPE =================
function startCamera() {
  const holistic = new Holistic({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}` });
  holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  holistic.onResults(res => { if (isRecording) recordedFrames.push(extract76Keypoints(res)); });

  const camera = new Camera(videoElement, {
    onFrame: async () => { await holistic.send({ image: videoElement }); },
    width: 640, height: 480
  });
  camera.start().catch(err => {
    statusEl.textContent = "AI đã sẵn sàng nhưng không mở được Camera: " + err.message;
  });
}

recordBtn.addEventListener("click", () => {
  recordedFrames = [];
  isRecording = true;
  recordBtn.disabled = true;
  statusEl.textContent = "Đang ghi hình (3 giây)...";
  setTimeout(async () => {
    isRecording = false;
    await evaluateSign();
    recordBtn.disabled = false;
  }, 3000);
});

init();