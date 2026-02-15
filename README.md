# Experiment 4: LoGoMIL for PI-CAI (Prostate Cancer Detection)

This repository implements **Experiment 4** of the thesis: a **Local–Global Multiple Instance Learning (LoGoMIL)** model for classifying prostate MRI studies as clinically significant prostate cancer (csPCa) positive or negative, using the PI-CAI dataset. The local branch is initialized from **patch-pretrained** weights; the full model is trained with early stopping and epoch-level logging.

---

## Overview

- **Task:** Binary classification (csPCa YES/NO) at the **study level** (one label per patient/study).
- **Inputs:**
  - **Local:** A *bag* of 3D patches (e.g. 8×8×8) per study — variable number of patches per case.
  - **Global:** A single 3D volume (32×64×64) per study (whole prostate).
- **Model:** LoGoMIL = patch encoder + attention pooling (local) + global encoder; features are summed and passed through a classifier.
- **Training:** One fold at a time (e.g. `data.fold=0`); validation AUC used for checkpointing and early stopping. Test fold is held out.

---

## Project Structure

```
Experiment4/
├── configs/
│   ├── config.yaml          # Main config (composes data, model, train, wandb)
│   ├── data/
│   │   └── picai.yaml       # Data paths, folds, val_ratio, label column
│   ├── model/
│   │   └── dual_branch.yaml # LoGoMIL: in_channels, emb_dim, attn_dim, etc.
│   ├── train/
│   │   └── default.yaml     # LR, batch size, epochs, scheduler, early stopping, exp4_patch
│   └── wandb/
│       └── default.yaml    # W&B project, entity, mode (e.g. offline)
├── src/
│   ├── data.py              # Manifests, CV splits, PicaiBagsDataset, collate_bags
│   ├── model.py             # IlseAttention, LocalLeNet3D, GlobalLeNet3D, LoGoMIL
│   ├── train.py             # Main training script (with patch-pretrained local + early stop)
│   ├── train_old.py         # Older training script (no patch pretrain / different options)
│   ├── eval_val.py          # Evaluate best checkpoint on validation set + ROC/confusion plots
│   └── eval_cv_val_auc.py   # Run all folds, compute mean±std validation AUC, plot CV ROC
├── submit_exp4_train.slurm # SLURM job to train one fold (set FOLD=0..4)
├── outputs/                 # Hydra run dirs (e.g. outputs/exp4_logo/fold0)
└── logs/                    # SLURM stdout/stderr
```

---

## How the Code Works

### 1. Data pipeline (`src/data.py`)

- **Manifests:** For each fold `i`, the code expects:
  - `{bags_base}/fold{i}/manifest.csv`
  - Columns: at least a bag path column (`bag_npz`, `bag_path`, `npz_path`, or `path`), and preferably `patient_id` / `study_id`.
- **Labels:** If the manifest does not already have a `label` column, labels are merged from a **marksheet** CSV (e.g. `case_csPCa` YES/NO → 0/1) using `study_id` or `patient_id`.
- **Splits:** `make_cv_splits_from_manifests()`:
  - Uses one fold as **test** (held out).
  - From the remaining folds, does a **stratified** train/val split with `val_ratio` (e.g. 10%).
- **PicaiBagsDataset:** For each row:
  - Loads a bag from an NPZ: key `"patches"` → shape `(K, 8, 8, 8)` → `x_local` (K, 1, 8, 8, 8).
  - Loads global volume from `{global_root}/{patient_id}_{study_id}.npy` → (32, 64, 64) → `x_global` (1, 32, 64, 64).
  - Returns `(x_local, x_global, y, K)`.
- **collate_bags:** Pads local bags to the same K in the batch and builds a boolean `mask` (True = real patch, False = padding) for attention.

### 2. Model (`src/model.py`)

- **IlseAttention:** Attention-based MIL pooling (Ilse et al.). Input `H` (B, K, M) → tanh(V(H)), then `w` → scores → softmax → weighted sum over K → bag embedding (B, M). Uses `mask` to ignore padded instances.
- **LocalLeNet3D:** Small 3D CNN for patches (1, 8, 8, 8) → conv layers + maxpool → FC → 256-d embedding per patch.
- **GlobalLeNet3D:** 3D CNN for full volume (1, 32, 64, 64) → conv + pool layers → FC → 256-d embedding per study.
- **LoGoMIL:**
  - Local: patches → `LocalLeNet3D` → (B, K, 256) → `IlseAttention` → (B, 256).
  - Global: volume → `GlobalLeNet3D` → (B, 256).
  - Fuse: `z = z_local + z_global`, then `classifier(z)` → (B,) logits.
  - **Experiment 4:** If `pretrained_local_path` is set, the **local encoder** is loaded from that checkpoint (`encoder_state` or `encoder.*` in state dict). The global encoder and rest of the model are trained from scratch.

### 3. Training (`src/train.py`)

- **Setup:** Hydra loads `configs/config.yaml` (and defaults for data, model, train, wandb). Seed, device, and optional W&B init.
- **Data:** Builds train/val/test splits and `PicaiBagsDataset` for train and val; DataLoaders use `collate_bags`.
- **Patch-pretrained local:** Reads `cfg.train.exp4_patch.pretrained_local_dir` and looks for `exp4_patch_best_fold{fold}.pt` for the current fold; passes this path to `LoGoMIL(..., pretrained_local_path=...)`.
- **Optimization:** Adam (LR and weight decay from config), ReduceLROnPlateau on validation loss.
- **Loop:** Each epoch: train on train set (BCEWithLogitsLoss), then evaluate on val set (loss + AUC). Logs at **epoch level** (e.g. to W&B). Saves last checkpoint every epoch; saves **best** checkpoint when validation AUC improves.
- **Early stopping:** After a warmup of `warmup` epochs, if validation AUC does not improve for `patience` epochs (within `min_delta`), training stops.
- **Resume:** If `last_fold{fold}.pt` exists, training resumes from it (model, optimizer, scheduler, epoch, best_auc, epochs_no_improve).

### 4. Evaluation

- **eval_val.py:** Loads `best_fold{fold}.pt`, runs validation set through the model, computes loss, AUC, ROC curve, and confusion matrix at threshold 0.5. Writes:
  - `val_predictions_fold{fold}.csv` (per-case predictions),
  - `val_roc_fold{fold}.png`,
  - `val_confusion_fold{fold}.png`.
  Note: `eval_val.py` builds `LoGoMIL` **without** `pretrained_local_path` and loads full `model_state` from the best checkpoint (the checkpoint already contains the trained local encoder).

- **eval_cv_val_auc.py:** For each fold 0..num_folds-1:
  - Finds best LoGoMIL checkpoint (`best_fold{fold}.pt`) and patch-pretrain checkpoint for that fold.
  - Builds LoGoMIL **with** `pretrained_local_path` and loads LoGoMIL `model_state` (so architecture matches what was trained).
  - Runs validation set, computes AUC and ROC.
  - Aggregates: mean and std of validation AUC, writes `cv_val_auc_summary_exp4.csv` and `cv_val_roc_allfolds_exp4.png`.

---

## Configuration

- **Data:** Set `data.bags_base`, `data.global_root`, `data.marksheet`, `data.fold`, `data.num_folds`, `data.val_ratio`, `data.label_col` in config or CLI.
- **Model:** `model.in_channels`, `model.emb_dim`, `model.attn_dim` (and any overrides in `configs/model/`).
- **Train:** `train.batch_size`, `train.epochs`, `train.lr`, `train.scheduler.*`, `train.early_stopping.*`, and `train.exp4_patch.pretrained_local_dir` for patch-pretrained local encoder path.
- **Override from CLI:** e.g. `python -m src.train data.fold=2 data.bags_base=/path/to/bags`.

---

## How to Run

### Prerequisites

- Python 3.x with PyTorch, Hydra (OmegaConf), pandas, numpy, scikit-learn, matplotlib.
- Data: bag NPZ files per fold manifest, global `.npy` volumes, and (if needed) marksheet CSV.

### Train one fold (local)

```bash
python -m src.train data.fold=0 hydra.run.dir=outputs/exp4_logo/fold0
```

Ensure `train.exp4_patch.pretrained_local_dir` points to the directory containing `exp4_patch_best_fold0.pt` (and similarly for other folds), or the script will raise `FileNotFoundError`.

### Train one fold (SLURM)

```bash
mkdir -p logs
sbatch --export=FOLD=0 submit_exp4_train.slurm   # fold 0
sbatch --export=FOLD=1 submit_exp4_train.slurm   # fold 1
# ... FOLD=2,3,4
```

The script sets `hydra.run.dir=outputs/exp4_logo/fold${FOLD}` and runs `python -m src.train data.fold=${FOLD} ...`.

### Evaluate best checkpoint on validation set

```bash
python -m src.eval_val data.fold=0
```

Writes validation metrics and plots; expects `best_fold0.pt` (and config) in the current run directory.

### Cross-validation summary (all folds)

From a directory where config is available and best checkpoints for each fold can be found (e.g. project root or outputs):

```bash
python -m src.eval_cv_val_auc
```

Produces `cv_val_auc_summary_exp4.csv` and `cv_val_roc_allfolds_exp4.png` with mean ± std validation AUC.

---

## Checkpoints

- **Training:** `last_fold{f}.pt` (latest epoch), `best_fold{f}.pt` (best validation AUC).
- **Patch pretrain (Experiment 4):** `exp4_patch_best_fold{f}.pt` — used only to initialize the **local** encoder; path is configured via `train.exp4_patch.pretrained_local_dir`.

---

## Notes

- The **test** fold is never used during training or validation; it is reserved for final reporting.
- W&B logging is optional; disable in `configs/wandb/default.yaml` or set `wandb.enabled=false`.
- Paths in the example configs point to cluster paths (`/scratch/...`, `/rhea/...`); adjust for your environment.
