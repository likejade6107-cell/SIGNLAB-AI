import os
import imageio.v2 as imageio
from tqdm import tqdm

def main():
    vid_dir = "web/videos"
    files = [f for f in os.listdir(vid_dir) if f.endswith(".mp4")]
    print(f"Đang chuẩn hóa {len(files)} video sang chuẩn H.264 cho trình duyệt...")

    for fname in tqdm(files):
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
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print("Hoàn tất chuẩn hóa! Tất cả video đã sẵn sàng phát trên mọi trình duyệt.")

if __name__ == "__main__":
    main()