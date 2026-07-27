"""
Leave-One-Study-Out (LOSO) Cross-Validation for Spaceflight cVAE
==================================================================
Holds out one entire study at a time as the validation set, so no
sample from that study is ever seen during training for that fold.
This tests whether the flight/ground classifier (and the encoder
that feeds it) generalizes to a genuinely new study, rather than
partially relying on study-specific batch effects — a real risk
when fine-tuning is pooled across multiple GeneLab-style datasets.

Each fold:
  1. Builds a FRESH model (reloading --pretrain_checkpoint if given,
     so no weights leak between folds).
  2. Trains on all studies except the held-out one.
  3. Validates only on the held-out study.
  4. Saves its own checkpoint under <output_dir>/loso_<study>/best_model.pt

At the end, prints per-fold and aggregate (mean +/- std) val_loss,
val_acc, and val_auroc across folds. Folds where the held-out study
has only one class (all flight or all ground) are skipped for AUROC
purposes since it's undefined, but still trained/logged.

Usage:
    python loso_cv.py --data genelab_finetune.h5 \\
        --pretrain_checkpoint pretrain_best.pt \\
        --epochs 100 --output_dir ./checkpoints/loso
"""

import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import SpaceflightDataset
from train import build_arg_parser, run_training


def run_loso_cv(args):
    print("Loading data...")
    dataset = SpaceflightDataset(args.data)

    fold_results = []

    for study_name, train_subset, val_subset in dataset.loso():
        # Skip folds where the held-out study has only one class present —
        # classification metrics (esp. AUROC) aren't meaningful there.
        val_indices = val_subset.indices
        val_flight = dataset.flight[val_indices]
        if len(np.unique(val_flight)) < 2:
            print(f"[{study_name}] SKIPPED — held-out study has only one "
                  f"class (n={len(val_indices)}, "
                  f"flight={int((val_flight==1).sum())}, "
                  f"ground={int((val_flight==0).sum())})")
            continue

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        safe_name = study_name.replace("/", "_").replace(" ", "_")
        ckpt_path = Path(args.output_dir) / f"loso_{safe_name}" / "best_model.pt"
        run_label = f"loso_{safe_name}"

        metrics = run_training(
            args, dataset, train_loader, val_loader, ckpt_path, run_label=run_label
        )
        metrics["study"] = study_name
        metrics["n_val"] = len(val_indices)
        fold_results.append(metrics)

    # --- Summary across folds ---
    print("\n" + "=" * 60)
    print("LOSO CV Summary")
    print("=" * 60)
    for r in fold_results:
        print(f"  {r['study']:<20} n_val={r['n_val']:<5} "
              f"val_loss={r['val_loss']:.4f}  "
              f"val_acc={r['val_acc']:.3f}  "
              f"val_auroc={r['val_auroc']:.3f}")

    if fold_results:
        accs   = [r["val_acc"] for r in fold_results]
        aurocs = [r["val_auroc"] for r in fold_results]
        losses = [r["val_loss"] for r in fold_results]
        print("-" * 60)
        print(f"  {'MEAN +/- STD':<20} "
              f"val_loss={np.mean(losses):.4f}+/-{np.std(losses):.4f}  "
              f"val_acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}  "
              f"val_auroc={np.nanmean(aurocs):.3f}+/-{np.nanstd(aurocs):.3f}")

        # Flag folds that look like outliers relative to the mean —
        # worth inspecting individually (see the batch-effect discussion:
        # a single bad fold often means that study behaves very differently).
        acc_mean, acc_std = np.mean(accs), np.std(accs)
        for r in fold_results:
            if acc_std > 0 and (r["val_acc"] < acc_mean - 1.5 * acc_std):
                print(f"  \u26a0 '{r['study']}' is a low outlier "
                      f"(acc={r['val_acc']:.3f} vs mean={acc_mean:.3f}) "
                      f"— worth inspecting for batch effects or mislabeling")
    else:
        print("  No folds were run (all studies skipped — check labels).")

    return fold_results


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_loso_cv(args)
