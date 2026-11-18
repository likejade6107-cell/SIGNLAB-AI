import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import VSLDataset
from model import Conv1DBiGRUScorer

def main():
    with open("models/label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    inv_map = {v: k for k, v in label_map.items()}

    model = Conv1DBiGRUScorer(num_classes=len(label_map))
    model.load_state_dict(torch.load("models/vsl_best.pth", map_location="cpu"))
    model.eval()

    # 1. Xuất sang ONNX cho trình duyệt
    dummy_input = torch.randn(1, 64, 228)
    torch.onnx.export(
        model,
        dummy_input,
        "models/vsl_scorer.onnx",
        input_names=["keypoints"],
        output_names=["logits", "embedding"],
        dynamic_axes={"keypoints": {0: "batch_size"}},
        opset_version=14
    )
    print("Đã xuất models/vsl_scorer.onnx")

    # 2. Tính vector trung bình chuẩn (centroid embedding) cho từng từ để chấm điểm
    train_ds = VSLDataset("data", split="train", max_len=64, label_map=label_map)
    loader = DataLoader(train_ds, batch_size=64, shuffle=False)

    embeds_by_class = {i: [] for i in range(len(label_map))}
    with torch.no_grad():
        for x, y in loader:
            _, emb = model(x)
            for e, label in zip(emb.numpy(), y.numpy()):
                embeds_by_class[int(label)].append(e)

    ref_embeddings = {}
    for idx, emb_list in embeds_by_class.items():
        mean_emb = np.mean(emb_list, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        ref_embeddings[inv_map[idx]] = mean_emb.tolist()

    with open("models/reference_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(ref_embeddings, f, ensure_ascii=False)
    print("Đã xuất models/reference_embeddings.json")

if __name__ == "__main__":
    main()