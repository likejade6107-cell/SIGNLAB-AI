import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset import VSLDataset
from model import Conv1DBiGRUScorer

# Đảm bảo hiển thị font tiếng Việt không lỗi trên đồ thị
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def evaluate_scoring_metrics(embeddings, targets, num_pairs=10000):
    """Tính các metric chuyên biệt cho bài toán chấm điểm (Embedding Similarity & ROC-AUC/EER)"""
    rng = np.random.default_rng(42)
    N = len(targets)
    
    # Nhóm index theo từng lớp từ vựng
    class_to_indices = {}
    for idx, label in enumerate(targets):
        class_to_indices.setdefault(int(label), []).append(idx)
    valid_classes = [c for c, idxs in class_to_indices.items() if len(idxs) >= 2]

    intra_sims = []  # Cùng một từ (Positive pairs)
    inter_sims = []  # Khác từ (Negative pairs)

    # Lấy mẫu cặp cùng lớp (Intra-class)
    for _ in range(num_pairs):
        c = rng.choice(valid_classes)
        i, j = rng.choice(class_to_indices[c], size=2, replace=False)
        sim = float(np.dot(embeddings[i], embeddings[j]))
        intra_sims.append(sim)

    # Lấy mẫu cặp khác lớp (Inter-class)
    all_classes = list(class_to_indices.keys())
    for _ in range(num_pairs):
        c1, c2 = rng.choice(all_classes, size=2, replace=False)
        i = rng.choice(class_to_indices[c1])
        j = rng.choice(class_to_indices[c2])
        sim = float(np.dot(embeddings[i], embeddings[j]))
        inter_sims.append(sim)

    intra_sims = np.array(intra_sims)
    inter_sims = np.array(inter_sims)

    # Tính ROC-AUC và EER (Equal Error Rate)
    y_true = np.concatenate([np.ones_like(intra_sims), np.zeros_like(inter_sims)])
    y_scores = np.concatenate([intra_sims, inter_sims])
    auc_score = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float(fpr[eer_idx])
    eer_threshold = float(thresholds[eer_idx])

    return {
        "intra_sim_mean": float(np.mean(intra_sims)),
        "intra_sim_std": float(np.std(intra_sims)),
        "inter_sim_mean": float(np.mean(inter_sims)),
        "inter_sim_std": float(np.std(inter_sims)),
        "roc_auc": float(auc_score),
        "eer": eer,
        "optimal_score_threshold": eer_threshold,
        "intra_sims": intra_sims,
        "inter_sims": inter_sims,
        "fpr": fpr,
        "tpr": tpr
    }


def plot_all_charts(history, scoring_res, all_targets, all_preds, inv_label_map, save_dir="models"):
    # 1. Đồ thị Loss & Accuracy qua các Epoch
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs_range, history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Loss qua các Epoch (Conv1D + BiGRU)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs_range, history["train_acc"], label="Train Accuracy (%)", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs_range, history["val_acc"], label="Validation Accuracy (%)", color="#d62728", linewidth=2)
    axes[1].set_title("Độ chính xác Top-1 qua các Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300)
    plt.close()

    # 2. Đồ thị Phân phối Cosine Similarity & Đường cong ROC-AUC
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.kdeplot(scoring_res["intra_sims"], ax=axes[0], fill=True, color="#2ca02c", label="Cùng từ (Intra-class)")
    sns.kdeplot(scoring_res["inter_sims"], ax=axes[0], fill=True, color="#d62728", label="Khác từ (Inter-class)")
    axes[0].axvline(scoring_res["optimal_score_threshold"], color="black", linestyle="--",
                    label=f"Ngưỡng tối ưu ({scoring_res['optimal_score_threshold']:.2f})")
    axes[0].set_title("Phân phối Cosine Similarity (Chấm điểm động tác)")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Mật độ phân phối")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(scoring_res["fpr"], scoring_res["tpr"], color="#1f77b4", lw=2,
                 label=f"ROC Curve (AUC = {scoring_res['roc_auc']:.4f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set_title("Đường cong ROC xác thực động tác người học")
    axes[1].set_xlabel("False Positive Rate (Chấm nhầm đạt)")
    axes[1].set_ylabel("True Positive Rate (Chấm đúng đạt)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "similarity_and_roc.png"), dpi=300)
    plt.close()

    # 3. Top 15 cặp từ hay bị nhầm lẫn nhất (trích từ Confusion Matrix 472x472)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(inv_label_map))))
    np.fill_diagonal(cm, 0)  # Bỏ qua các dự đoán đúng trên đường chéo chính
    flat_indices = np.argsort(cm.ravel())[::-1][:15]
    rows, cols = np.unravel_index(flat_indices, cm.shape)

    pair_labels = []
    pair_counts = []
    for r, c in zip(rows, cols):
        if cm[r, c] > 0:
            pair_labels.append(f"{inv_label_map[r]} -> {inv_label_map[c]}")
            pair_counts.append(int(cm[r, c]))

    if len(pair_counts) > 0:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=pair_counts, y=pair_labels, palette="Reds_r")
        plt.title("Top các cặp từ hay bị nhầm lẫn nhất (Thực tế -> Dự đoán)")
        plt.xlabel("Số lượng mẫu bị nhầm")
        plt.ylabel("Cặp từ vựng")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "top_confused_pairs.png"), dpi=300)
        plt.close()


def main():
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {device}")

    # 1. Tải dữ liệu và chia 3 tập Train / Validation / Test
    full_train_ds = VSLDataset("data", split="train", max_len=64)
    test_ds = VSLDataset("data", split="test", max_len=64, label_map=full_train_ds.label_map)
    inv_label_map = {v: k for k, v in full_train_ds.label_map.items()}

    labels = [full_train_ds.label_map[os.path.basename(os.path.dirname(f))] for f in full_train_ds.files]
    train_idx, val_idx = train_test_split(
        range(len(full_train_ds)), test_size=0.15, random_state=42, stratify=labels
    )

    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_train_ds, val_idx)
    print(f"Số mẫu -> Train: {len(train_ds)} | Validation: {len(val_ds)} | Test: {len(test_ds)}")

    with open("models/label_map.json", "w", encoding="utf-8") as f:
        json.dump(full_train_ds.label_map, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    model = Conv1DBiGRUScorer(num_classes=len(full_train_ds.label_map)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    epochs = 35

    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total * 100

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total * 100
        scheduler.step(val_acc)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/vsl_best.pth")
            print(f"--> Đã lưu model tốt nhất ({best_val_acc:.2f}%)")

    # --- ĐÁNH GIÁ TOÀN DIỆN TRÊN TẬP TEST ĐỘC LẬP ---
    print("\nĐang tính toán toàn bộ chỉ số đánh giá và xuất biểu đồ...")
    model.load_state_dict(torch.load("models/vsl_best.pth", map_location=device))
    model.eval()

    all_preds, all_targets, all_embeddings = [], [], []
    top1_correct, top5_correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, emb = model(x)

            preds = torch.argmax(logits, dim=1)
            top1_correct += (preds == y).sum().item()

            _, top5_preds = logits.topk(5, dim=1)
            top5_correct += (top5_preds == y.unsqueeze(1)).any(dim=1).sum().item()

            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    scoring_res = evaluate_scoring_metrics(all_embeddings, all_targets)

    report = {
        "classification_metrics": {
            "top1_accuracy_percent": round(top1_correct / total * 100, 2),
            "top5_accuracy_percent": round(top5_correct / total * 100, 2),
            "macro_precision_percent": round(precision * 100, 2),
            "macro_recall_percent": round(recall * 100, 2),
            "macro_f1_percent": round(f1 * 100, 2),
        },
        "scoring_embedding_metrics": {
            "intra_class_cosine_sim_mean": round(scoring_res["intra_sim_mean"], 4),
            "intra_class_cosine_sim_std": round(scoring_res["intra_sim_std"], 4),
            "inter_class_cosine_sim_mean": round(scoring_res["inter_sim_mean"], 4),
            "inter_class_cosine_sim_std": round(scoring_res["inter_sim_std"], 4),
            "roc_auc": round(scoring_res["roc_auc"], 4),
            "equal_error_rate_eer": round(scoring_res["eer"], 4),
            "optimal_similarity_threshold": round(scoring_res["optimal_score_threshold"], 4),
        }
    }

    with open("models/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    plot_all_charts(history, scoring_res, all_targets, all_preds, inv_label_map, save_dir="models")

    print("\n================ BÁO CÁO ĐÁNH GIÁ MÔ HÌNH ================")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("==========================================================")
    print("Đã lưu 3 biểu đồ trực quan hóa vào thư mục 'models/':")
    print("1. models/training_curves.png")
    print("2. models/similarity_and_roc.png")
    print("3. models/top_confused_pairs.png")


if __name__ == "__main__":
    main()