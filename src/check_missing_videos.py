import os
import json

def main():
    with open("models/label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)

    vid_dir = "web/videos"
    missing_words = []

    for word in sorted(label_map.keys()):
        safe_name = word.replace("/", "_").replace("\\", "_").replace("?", "")
        file_path = os.path.join(vid_dir, f"{safe_name}.mp4")

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            missing_words.append(word)

    print(f"Tổng số từ thiếu video: {len(missing_words)} / {len(label_map)}\n")
    for idx, word in enumerate(missing_words, start=1):
        print(f"{idx:2d}. {word}")

    # Lưu ra file txt để tiện đối chiếu hoặc tìm tải bổ sung
    with open("missing_videos.txt", "w", encoding="utf-8") as f:
        for word in missing_words:
            f.write(word + "\n")
    print("\nĐã lưu danh sách ra file 'missing_videos.txt'.")

if __name__ == "__main__":
    main()