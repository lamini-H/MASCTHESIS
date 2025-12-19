import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import make_cv_splits, PicaiDualBranchBags, collate_dual_branch
from src.model import DualBranchAttentionNet

# -------------------------
# Utils
# -------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # determinism tradeoff; keep fast but stable-ish
    torch.backends.cudnn.benchmark = True

def binary_auc_from_logits(all_logits: np.ndarray, all_targets: np.ndarray) -> float:
    """
    Compute ROC AUC for binary classification.
    Uses sklearn if available; otherwise fallback to a simple rank-based AUC.
    """
    probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid on logit for class=1
    y = all_targets.astype(np.int64)

    # if only one class present, AUC is undefined
    if len(np.unique(y)) < 2:
        return float("nan")

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, probs))
    except Exception:
        # Rank-based AUC fallback (Mann–Whitney U)
        order = np.argsort(probs)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(probs), dtype=np.float64) + 1.0

        pos = (y == 1)
        n_pos = pos.sum()
        n_neg = (~pos).sum()
        sum_ranks_pos = ranks[pos].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auc)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_n = 0

    all_logits = []
    all_targets = []

    loss_fn = nn.CrossEntropyLoss()

    for x_local, x_global, y in loader:
        x_local = x_local.to(device)     # (B,K,1,16,32,32)
        x_global = x_global.to(device)   # (B,1,32,64,64)
        y = y.to(device)                 # (B,)

        logits = model(x_local, x_global)  # (B,2)
        loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        # take logit for class 1 for AUC
        all_logits.append(logits[:, 1].detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    avg_loss = total_loss / max(1, total_n)
    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    auc = binary_auc_from_logits(all_logits, all_targets)
    return avg_loss, auc

# -------------------------
# Main train loop
# -------------------------
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("=== CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    seed_all(int(cfg.train.seed))

    device = torch.device("cuda" if (cfg.train.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device: {device}")

    # --------- Load CSV and make CV splits (test fold held out) ---------
    df = pd.read_csv(cfg.data.csv_path)

    # IMPORTANT: create test_df but DO NOT use it in training job
    train_df, val_df, test_df = make_cv_splits(
        df=df,
        local_root=cfg.data.local_root,
        num_folds=int(cfg.data.num_folds),
        test_fold=int(cfg.data.fold),
        val_ratio=float(cfg.data.val_ratio),
        seed=int(cfg.data.seed),
    )

    print(f"[FOLD {cfg.data.fold}] train={len(train_df)} val={len(val_df)} test(HELD-OUT)={len(test_df)}")

    # --------- Datasets / Loaders (train+val only) ---------
    train_set = PicaiDualBranchBags(
        df=train_df,
        local_root=cfg.data.local_root,
        global_root=cfg.data.global_root,
        num_folds=int(cfg.data.num_folds),
        k_patches=int(cfg.data.k_patches),
        sample_strategy=str(cfg.data.sample_strategy),
        pad_value=float(cfg.data.pad_value),
        label_col=str(cfg.data.label_col),
    )

    val_set = PicaiDualBranchBags(
        df=val_df,
        local_root=cfg.data.local_root,
        global_root=cfg.data.global_root,
        num_folds=int(cfg.data.num_folds),
        k_patches=int(cfg.data.k_patches),
        sample_strategy="first",  # deterministic for validation
        pad_value=float(cfg.data.pad_value),
        label_col=str(cfg.data.label_col),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_dual_branch,
        drop_last=False,
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

    # --------- Model ---------
    model = DualBranchAttentionNet(
        in_channels=int(cfg.model.in_channels),
        base_channels=int(cfg.model.base_channels),
        dropout=float(cfg.model.dropout),
        emb_dim=int(cfg.model.fusion_dim),   # reuse fusion_dim as embedding size
        attn_dim=128,
        num_classes=int(cfg.model.num_classes),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    loss_fn = nn.CrossEntropyLoss()

    # --------- Training ---------
    best_auc = -1.0
    best_path = os.path.abspath(f"best_fold{cfg.data.fold}.pt")

    global_step = 0
    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch_idx, (x_local, x_global, y) in enumerate(train_loader, start=1):
            x_local = x_local.to(device)
            x_global = x_global.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_local, x_global)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            running_loss += loss.item() * bs
            seen += bs
            global_step += 1

            if global_step % int(cfg.train.log_every) == 0:
                print(f"[epoch {epoch} step {global_step}] train_loss={running_loss/max(1,seen):.4f}")

        train_loss = running_loss / max(1, seen)
        val_loss, val_auc = evaluate(model, val_loader, device)

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_auc={val_auc:.4f}")

        # save best by val AUC
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            ckpt = {
                "epoch": epoch,
                "best_auc": best_auc,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }
            torch.save(ckpt, best_path)
            print(f"[SAVE] new best val_auc={best_auc:.4f} -> {best_path}")

    print(f"[DONE] Best val AUC for fold {cfg.data.fold}: {best_auc:.4f}")
    print("[NOTE] Test fold is held out and was NOT used. Run a separate eval job if needed.")

if __name__ == "__main__":
    main()
