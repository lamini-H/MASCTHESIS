import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import make_cv_splits, PicaiDualBranchBags, collate_dual_branch
from src.model import DualBranchAttentionNet

import matplotlib.pyplot as plt


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_auc_roc(y_true, y_prob):
    """
    Returns (auc, fpr, tpr, thresholds).
    Uses sklearn if available; else uses a simple fallback for AUC + ROC points.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return float("nan"), None, None, None

    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        return auc, fpr, tpr, thr
    except Exception:
        # Fallback AUC (rank-based); ROC curve not provided
        order = np.argsort(y_prob)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(y_prob), dtype=np.float64) + 1.0

        pos = (y_true == 1)
        n_pos = pos.sum()
        n_neg = (~pos).sum()
        sum_ranks_pos = ranks[pos].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auc), None, None, None


def confusion_from_probs(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    # TN, FP, FN, TP
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp, y_pred


@torch.no_grad()
def run_validation(model, val_df, cfg, device):
    val_set = PicaiDualBranchBags(
        df=val_df,
        local_root=cfg.data.local_root,
        global_root=cfg.data.global_root,
        num_folds=int(cfg.data.num_folds),
        k_patches=int(cfg.data.k_patches),
        sample_strategy="first",  # deterministic validation
        pad_value=float(cfg.data.pad_value),
        label_col=str(cfg.data.label_col),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_dual_branch,
        drop_last=False,
    )

    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    all_logits1 = []
    all_targets = []
    all_ids = []

    total_loss = 0.0
    total_n = 0

    for i, (x_local, x_global, y) in enumerate(val_loader):
        x_local = x_local.to(device)
        x_global = x_global.to(device)
        y = y.to(device)

        logits = model(x_local, x_global)  # (B,2)
        loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        all_logits1.append(logits[:, 1].detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

        # pull IDs from val_df rows corresponding to this mini-batch index range
        # DataLoader is sequential, so indices map directly.
        start = i * int(cfg.train.batch_size)
        end = min(start + bs, len(val_df))
        sub = val_df.iloc[start:end]
        all_ids.extend(list(zip(sub["patient_id"].tolist(), sub["study_id"].tolist())))

    logits1 = np.concatenate(all_logits1, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_prob = sigmoid(logits1)

    val_loss = total_loss / max(1, total_n)
    auc, fpr, tpr, thr = compute_auc_roc(y_true, y_prob)

    return val_loss, auc, fpr, tpr, y_true, y_prob, all_ids


def plot_roc(fpr, tpr, auc, out_path):
    plt.figure(figsize=(6, 5))
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Validation ROC")
        plt.legend()
    else:
        plt.title(f"Validation ROC (AUC={auc:.4f})\nROC curve unavailable (no sklearn fallback)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrix(tn, fp, fn, tp, out_path):
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=int)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["0 (NO)", "1 (YES)"])
    plt.yticks([0, 1], ["0 (NO)", "1 (YES)"])

    # annotate
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    seed_all(int(cfg.train.seed))

    device = torch.device("cuda" if (cfg.train.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device: {device}")

    # Load CSV and create CV splits for this fold
    df = pd.read_csv(cfg.data.csv_path)
    train_df, val_df, test_df = make_cv_splits(
        df=df,
        local_root=cfg.data.local_root,
        num_folds=int(cfg.data.num_folds),
        test_fold=int(cfg.data.fold),
        val_ratio=float(cfg.data.val_ratio),
        seed=int(cfg.data.seed),
    )

    print(f"[FOLD {cfg.data.fold}] train={len(train_df)} val={len(val_df)} test(HELD-OUT)={len(test_df)}")

    # Load best checkpoint for this fold
    ckpt_path = os.path.abspath(f"best_fold{cfg.data.fold}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = DualBranchAttentionNet(
        in_channels=int(cfg.model.in_channels),
        base_channels=int(cfg.model.base_channels),
        dropout=float(cfg.model.dropout),
        emb_dim=int(cfg.model.fusion_dim),
        attn_dim=128,
        num_classes=int(cfg.model.num_classes),
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)

    # Run validation eval
    val_loss, val_auc, fpr, tpr, y_true, y_prob, ids = run_validation(model, val_df, cfg, device)

    print(f"[VAL] loss={val_loss:.4f} auc={val_auc:.4f}")

    # Save per-case CSV
    out_csv = os.path.abspath(f"val_predictions_fold{cfg.data.fold}.csv")
    tn, fp, fn, tp, y_pred = confusion_from_probs(y_true, y_prob, threshold=0.5)

    rows = []
    for (pid, sid), yt, yp, yd in zip(ids, y_true, y_prob, y_pred):
        rows.append({
            "patient_id": pid,
            "study_id": sid,
            "y_true": int(yt),
            "prob_yes": float(yp),
            "y_pred": int(yd),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[SAVE] per-case val CSV -> {out_csv}")

    # Save plots
    roc_png = os.path.abspath(f"val_roc_fold{cfg.data.fold}.png")
    cm_png  = os.path.abspath(f"val_confusion_fold{cfg.data.fold}.png")

    plot_roc(fpr, tpr, val_auc, roc_png)
    plot_confusion_matrix(tn, fp, fn, tp, cm_png)

    print(f"[SAVE] ROC plot -> {roc_png}")
    print(f"[SAVE] Confusion matrix -> {cm_png}")
    print(f"[CM @0.5] TN={tn} FP={fp} FN={fn} TP={tp}")


if __name__ == "__main__":
    main()
