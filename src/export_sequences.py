import json
import numpy as np
from dataset import VSLDataset

def main():
    with open("models/label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)

    ds = VSLDataset("data", split="train", max_len=64, label_map=label_map)
    inv_map = {v: k for k, v in label_map.items()}
    
    sequences = {}
    for idx in range(len(ds)):
        x, y = ds[idx]
        word = inv_map[int(y)]
        if word not in sequences:
            # Làm tròn 3 chữ số thập phân để file JSON nhẹ (< 15MB cho 472 từ)
            seq = np.round(x.numpy().reshape(64, 76, 3), 3).tolist()
            sequences[word] = seq
        if len(sequences) == len(label_map):
            break

    with open("web/models/reference_sequences.json", "w", encoding="utf-8") as f:
        json.dump(sequences, f)
    print(f"Đã xuất chuỗi mẫu chuẩn cho {len(sequences)} từ!")

if __name__ == "__main__":
    main()