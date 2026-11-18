import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class VSLDataset(Dataset):
    def __init__(self, data_dir, split="train", max_len=64, label_map=None):
        self.max_len = max_len
        
        # Tự động dò tìm thư mục train / test bên trong data_dir (hoặc toàn bộ dự án)
        split_dir = None
        for search_root in [data_dir, "."]:
            for root, dirs, _ in os.walk(search_root):
                if os.path.basename(root) == split and len(dirs) > 0:
                    sample_npy = glob.glob(os.path.join(root, "*", "*.npy"))
                    if len(sample_npy) > 0:
                        split_dir = root
                        break
            if split_dir:
                break

        if split_dir is None:
            raise ValueError(f"Không tìm thấy thư mục '{split}' chứa file .npy. Hãy kiểm tra xem bạn đã giải nén vsl_keypoints.zip chưa.")

        self.files = glob.glob(os.path.join(split_dir, "*", "*.npy"))

        if label_map is None:
            classes = sorted(os.listdir(split_dir))
            self.label_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.files)

    def normalize(self, kp):
        T = kp.shape[0]
        mean = np.mean(kp, axis=(0, 1), keepdims=True)
        std = np.std(kp, axis=(0, 1), keepdims=True) + 1e-6
        kp = (kp - mean) / std
        return kp.reshape(T, -1)

    def pad_or_sample(self, seq):
        T, D = seq.shape
        if T >= self.max_len:
            idx = np.linspace(0, T - 1, self.max_len, dtype=int)
            return seq[idx]
        pad = np.zeros((self.max_len - T, D), dtype=np.float32)
        return np.vstack([seq, pad])

    def __getitem__(self, idx):
        path = self.files[idx]
        gloss = os.path.basename(os.path.dirname(path))
        label = self.label_map[gloss]

        kp = np.load(path).astype(np.float32)
        kp = self.normalize(kp)
        kp = self.pad_or_sample(kp)

        return torch.tensor(kp, dtype=torch.float32), torch.tensor(label, dtype=torch.long)