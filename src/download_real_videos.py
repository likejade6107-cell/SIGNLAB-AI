import os
import json
import time
import shutil
import subprocess
from tqdm import tqdm

DATASET = "nguyenanfms/vsl-vietnamese-sign-language-v2"
BASE_REMOTE_DIR = "processed/processed/frame_splited"

def safe_copy(src, dst, retries=5):
    for _ in range(retries):
        try:
            shutil.copy2(src, dst)
            return True
        except PermissionError:
            time.sleep(0.5)
    return False

def main():
    os.makedirs("web/videos", exist_ok=True)
    os.makedirs("temp_vid", exist_ok=True)

    with open("models/label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)

    with open("data/vsl_full_front_augmented.json", "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    word_to_meta = {}
    for item in meta_list:
        gloss = item.get("gloss", "").strip()
        vid_id = str(item.get("videoid", "")).strip()
        split = item.get("split", "test").strip()

        if gloss in label_map and vid_id.isdigit():
            if gloss not in word_to_meta or split == "test":
                word_to_meta[gloss] = (split, vid_id)

    print(f"Đã khớp {len(word_to_meta)}/{len(label_map)} từ. Bắt đầu tải & chuyển vào web/videos/...")

    success_count = 0
    for word, (split, vid_id) in tqdm(word_to_meta.items(), desc="Downloading VSL Videos"):
        safe_name = word.replace("/", "_").replace("\\", "_").replace("?", "")
        final_path = os.path.join("web", "videos", f"{safe_name}.mp4")

        if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
            success_count += 1
            continue

        # Kiểm tra xem file đã nằm sẵn trong temp_vid từ lần chạy trước chưa
        downloaded_path = None
        for root, _, files in os.walk("temp_vid"):
            if f"{vid_id}.mp4" in files:
                downloaded_path = os.path.join(root, f"{vid_id}.mp4")
                break

        # Nếu chưa có trong temp_vid thì tải mới từ Kaggle
        if not downloaded_path:
            remote_file = f"{BASE_REMOTE_DIR}/{split}/{word}/{vid_id}.mp4"
            cmd = [
                "kaggle", "datasets", "download",
                "-d", DATASET,
                "-f", remote_file,
                "-p", "temp_vid",
                "--unzip", "-q"
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            for root, _, files in os.walk("temp_vid"):
                if f"{vid_id}.mp4" in files:
                    downloaded_path = os.path.join(root, f"{vid_id}.mp4")
                    break

        if downloaded_path and os.path.exists(downloaded_path):
            if safe_copy(downloaded_path, final_path):
                success_count += 1

    print(f"\nHoàn tất! Đã có {success_count}/{len(label_map)} video người thật trong 'web/videos/'.")

if __name__ == "__main__":
    main()