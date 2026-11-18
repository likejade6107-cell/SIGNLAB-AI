import json

path = "web/models/reference_sequences.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Làm tròn tọa độ xuống 3 chữ số thập phân và loại bỏ khoảng trắng thừa
compressed = {
    word: [[[round(val, 3) for val in pt] for pt in frame] for frame in seq]
    for word, seq in data.items()
}

with open(path, "w", encoding="utf-8") as f:
    json.dump(compressed, f, ensure_ascii=False, separators=(",", ":"))

print("Đã nén xong reference_sequences.json!")