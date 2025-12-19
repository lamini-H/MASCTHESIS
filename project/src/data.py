import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def to_id_str(x):
    if pd.isna(x): return None
    try: return str(int(x))
    except Exception:
        s = str(x).strip()
        return s[:-2] if s.endswith(".0") else s

def locate_patient_fold_and_dir(local_root, num_folds, patient_id):
    for f in range(num_folds):
        pdir = os.path.join(local_root, f"instances_image_fold{f}", patient_id)
        if os.path.isdir(pdir):
            return f, pdir
    return None, None

def make_cv_splits(df, local_root, num_folds, test_fold, val_ratio=0.1, seed=42):
    """
    Returns three dataframes: train_df, val_df, test_df
    - test_df: patients whose local folder is in fold == test_fold
    - train/val: from remaining folds, split by val_ratio
    """
    # map each row -> fold (based on patient_id folder)
    folds = []
    keep = []
    for i, row in df.iterrows():
        pid = to_id_str(row["patient_id"])
        f, pdir = locate_patient_fold_and_dir(local_root, num_folds, pid)
        if f is None:
            folds.append(None)
            keep.append(False)
        else:
            folds.append(f)
            keep.append(True)

    df = df.copy()
    df["fold"] = folds
    df = df[keep].reset_index(drop=True)

    test_df = df[df["fold"] == test_fold].reset_index(drop=True)
    trainval_df = df[df["fold"] != test_fold].reset_index(drop=True)

    # deterministic shuffle then split
    rng = np.random.RandomState(seed)
    idx = np.arange(len(trainval_df))
    rng.shuffle(idx)

    n_val = int(round(val_ratio * len(trainval_df)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)

    return train_df, val_df, test_df

 # Custom Dataset class for Picai Dual Branch Bags
class PicaiDualBranchBags(Dataset):
    def __init__(self, df, local_root, global_root, num_folds=5,
                 k_patches=16, sample_strategy="random", pad_value=0.0,
                 label_col="label"):
        self.df = df.reset_index(drop=True)
        self.local_root = local_root
        self.global_root = global_root
        self.num_folds = num_folds
        self.k = int(k_patches)
        self.sample_strategy = sample_strategy
        self.pad_value = float(pad_value)
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def _load_patch(self, path):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = to_id_str(row["patient_id"])
        study_id   = to_id_str(row["study_id"])

        label_raw = str(row[self.label_col]).strip().upper()
        if label_raw == "YES":
            y = 1
        elif label_raw == "NO":
            y = 0
        else:
            raise ValueError(f"Unexpected label '{row[self.label_col]}' for {patient_id}_{study_id}")

        # GLOBAL
        gpath = os.path.join(self.global_root, f"{patient_id}_{study_id}.npy")
        x_global = np.load(gpath).astype(np.float32)
        if x_global.ndim == 3:
            x_global = x_global[None, ...]

        # LOCAL
        _, pdir = locate_patient_fold_and_dir(self.local_root, self.num_folds, patient_id)
        patch_paths = sorted(glob.glob(os.path.join(pdir, "patch_*.npy")))
        if len(patch_paths) == 0:
            raise FileNotFoundError(f"No patches in {pdir}")

        if len(patch_paths) >= self.k:
            if self.sample_strategy == "random":
                chosen = np.random.choice(patch_paths, size=self.k, replace=False)
                chosen = sorted(chosen)
            else:
                chosen = patch_paths[:self.k]
        else:
            chosen = patch_paths

        patches = np.stack([self._load_patch(p) for p in chosen], axis=0)  # (N,1,16,32,32)

        if patches.shape[0] < self.k:
            pad = np.full((self.k - patches.shape[0], 1, 16, 32, 32), self.pad_value, dtype=np.float32)
            patches = np.concatenate([patches, pad], axis=0)

        x_local  = torch.from_numpy(patches)          # (K,1,16,32,32)
        x_global = torch.from_numpy(x_global)         # (1,32,64,64)
        y = torch.tensor(y, dtype=torch.long)

        return x_local, x_global, y


# Collate function for Dual Branch Attention Network
def collate_dual_branch(batch):
    x_local  = torch.stack([b[0] for b in batch], dim=0)   # (B,K,1,16,32,32)
    x_global = torch.stack([b[1] for b in batch], dim=0)   # (B,1,32,64,64)
    y        = torch.stack([b[2] for b in batch], dim=0)
    return x_local, x_global, y
