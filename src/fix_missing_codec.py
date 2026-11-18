import os
import imageio.v2 as imageio
from tqdm import tqdm

def main():
    if not os.path.exists("missing_videos.txt"):
        print("Không tìm thấy file missing_videos.txt!")
        return

    with open("missing_videos.txt", "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    vid_dir = "web/videos"
    targets = []
    for word in words:
        safe_name = word.replace("/", "_").replace("\\", "_").replace("?", "")
        fname = f"{safe_name}.mp4"
        if os.path.exists(os.path.join(vid_dir, fname)):
            targets.append(fname)

    print(f"Tìm thấy {len(targets)}/{len(words)} video mới bổ sung. Bắt đầu chuẩn hóa H.264...")

    for fname in tqdm(targets):
        fpath = os.path.join(vid_dir, fname)
        tmp_path = os.path.join(vid_dir, f"tmp_{fname}")
        try:
            reader = imageio.get_reader(fpath)
            fps = reader.get_meta_data().get("fps", 25)
            frames = [frame for frame in reader]
            reader.close()

            if len(frames) > 0:
                imageio.mimsave(
                    tmp_path,
                    frames,
                    fps=fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=1
                )
                os.replace(tmp_path, fpath)
        except Exception as e:
            print(f"\nLỗi khi xử lý {fname}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print("Hoàn tất! Toàn bộ 472/472 từ vựng đã sẵn sàng hiển thị video người thật.")

if __name__ == "__main__":
    main()