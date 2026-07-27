"""
hpsearch.py — Optuna hyperparameter search for SpaceflightCVAE fine-tuning.

Wraps train.py's train() function directly so all training logic stays in one
place. Supports parallel SLURM workers via a shared SQLite study database.

Usage (single machine):
    python hpsearch.py --data subset_final.h5 --pretrain_dir /path/to/pretrain_checkpoints

Usage (parallel SLURM workers — run this command in N job scripts):
    python hpsearch.py --data subset_final.h5 --pretrain_dir /path/to/pretrain_checkpoints \
        --storage sqlite:///hpsearch.db --study_name spaceflight-cvae --n_trials 20

Stages:
    --stage arch   Search latent_dim, hidden_dims, conditions, emb_dims  (default)
    --stage loss   Fix arch from best trial, search lambda_adv/cls, beta, kl_anneal
    --stage train  Fix arch+loss, search lr, dropout, freeze_decoder, batch_size
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ---------------------------------------------------------------------------
# Import train() from train.py in the same directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from train import train as _train_fn   # the existing train(args) function

# Optional wandb — suppress per-trial wandb init noise during search;
# the search script logs its own summary at the end.
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pretrain checkpoint registry
# ---------------------------------------------------------------------------

def find_pretrain_checkpoint(pretrain_dir: str, latent_dim: int) -> str | None:
    """
    Look for a pretrain checkpoint matching the requested latent_dim.

    Expected naming convention (matches pretrain_archs4.py output):
        <pretrain_dir>/<latent_dim>_*/pretrain_best.pt
        e.g. /checkpoints/pretrain/64_32_filter_tissue/pretrain_best.pt

    Returns the path string if found, None otherwise.
    """
    pretrain_dir = Path(pretrain_dir)
    # Search for directories whose name starts with the latent_dim
    candidates = sorted(pretrain_dir.glob(f"{latent_dim}_*/pretrain_best.pt"))
    if candidates:
        return str(candidates[0])
    # Fallback: any pretrain_best.pt in a direct subdirectory
    candidates = sorted(pretrain_dir.glob("*/pretrain_best.pt"))
    if candidates:
        return str(candidates[0])
    return None


# ---------------------------------------------------------------------------
# Build an args namespace from trial suggestions + fixed overrides
# ---------------------------------------------------------------------------

def build_args(trial: optuna.Trial, base: argparse.Namespace,
               stage: str) -> SimpleNamespace:
    """
    Suggest hyperparameters for the given stage and merge with fixed base args.
    Returns a SimpleNamespace that train() accepts in place of argparse output.
    """
    args = SimpleNamespace(**vars(base))

    # ---- Stage 1: Architecture ------------------------------------------------
    if stage == "arch":
        args.latent_dim = trial.suggest_categorical("latent_dim", [32, 64, 128])

        # Hidden layer architecture: depth × width
        n_layers  = trial.suggest_int("n_layers", 2, 4)
        layer_dim = trial.suggest_categorical("layer_dim", [128, 256, 512])
        args.hidden_dims = [max(32, layer_dim // (2**i)) for i in range(n_layers)]

        # Condition subset — require at least one
        all_conds = ["tissue", "strain", "sex", "euth"]
        chosen = [c for c in all_conds
                  if trial.suggest_categorical(f"use_{c}", [True, False])]
        if not chosen:
            chosen = ["tissue"]
        args.conditions = chosen

        # Embedding dims — only for active conditions
        for cond in all_conds:
            dim_choices = [4, 8, 16, 32]
            suggested   = trial.suggest_categorical(f"{cond}_emb_dim", dim_choices)
            setattr(args, f"{cond}_emb_dim", suggested)

        # Keep loss weights fixed at sensible defaults during arch search
        args.lambda_adv     = 1.0
        args.lambda_cls     = 1.0
        args.beta           = 1.0
        args.kl_anneal_epochs = 100

    # ---- Stage 2: Loss weights ------------------------------------------------
    elif stage == "loss":
        # Architecture fixed — load from best_arch_params if provided
        args.lambda_adv       = trial.suggest_float("lambda_adv", 0.1, 5.0, log=True)
        args.lambda_cls       = trial.suggest_float("lambda_cls", 0.1, 2.0)
        args.beta             = trial.suggest_float("beta", 0.5, 2.0)
        args.kl_anneal_epochs = trial.suggest_int("kl_anneal_epochs", 50, 200, step=25)
        args.grl_alpha        = trial.suggest_float("grl_alpha", 0.1, 2.0)

    # ---- Stage 3: Training dynamics -------------------------------------------
    elif stage == "train":
        args.lr                   = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        args.new_lr_mult          = trial.suggest_categorical("new_lr_mult", [5.0, 10.0, 20.0])
        args.dropout              = trial.suggest_float("dropout", 0.1, 0.4)
        args.freeze_decoder_epochs = trial.suggest_int("freeze_decoder_epochs", 0, 30, step=5)
        args.batch_size           = trial.suggest_categorical("batch_size", [16, 32, 64])

    else:
        raise ValueError(f"Unknown stage: {stage!r}. Choose arch | loss | train")

    # Resolve pretrain checkpoint for the chosen latent_dim
    if base.pretrain_dir:
        ckpt = find_pretrain_checkpoint(base.pretrain_dir, args.latent_dim)
        if ckpt:
            args.pretrain_checkpoint = ckpt
        else:
            print(f"  [warn] No pretrain checkpoint found for latent_dim={args.latent_dim} "
                  f"in {base.pretrain_dir}. Training from scratch.")
            args.pretrain_checkpoint = None

    # Per-trial output directory so checkpoints don't collide
    args.output_dir = str(Path(base.output_dir) / f"trial_{trial.number:04d}")

    # Suppress wandb inside each trial — hpsearch logs its own summary
    args.no_wandb = True

    return args


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def make_objective(base_args: argparse.Namespace, stage: str):
    """
    Returns an Optuna objective function closed over base_args and stage.

    The objective trains one model and returns a composite score:
        score = val_loss + w_batch * study_acc_from_z - w_flight * val_auroc

    Lower is better (Optuna minimizes by default).
    """
    # Weights for the composite score — tune these to reflect priorities
    W_BATCH  = 3.0   # penalize high study predictability (batch effects remain)
    W_FLIGHT = 5.0   # reward high flight AUROC (biological signal preserved)

    def objective(trial: optuna.Trial) -> float:
        args = build_args(trial, base_args, stage)

        # Monkey-patch train() to also return the metrics we need.
        # We capture them via a mutable container since train() currently
        # prints but doesn't return values.
        results = {}

        original_train = _train_fn

        def patched_train(a):
            # Run the original training loop and intercept the final metrics
            # by temporarily replacing the checkpoint-save call.
            # Simpler: we re-implement the loop minimally here.
            _run_and_capture(a, results, trial)

        patched_train(args)

        val_loss   = results.get("best_val_loss", float("inf"))
        study_acc  = results.get("last_study_acc", 1.0)   # worst case: 1.0
        val_auroc  = results.get("best_val_auroc", 0.5)   # worst case: 0.5

        score = val_loss + W_BATCH * study_acc - W_FLIGHT * val_auroc

        # Log to trial user attributes for easy inspection later
        trial.set_user_attr("val_loss",   val_loss)
        trial.set_user_attr("study_acc",  study_acc)
        trial.set_user_attr("val_auroc",  val_auroc)
        trial.set_user_attr("score",      score)
        trial.set_user_attr("conditions", str(getattr(args, "conditions", [])))
        trial.set_user_attr("latent_dim", getattr(args, "latent_dim", -1))

        return score

    return objective


# ---------------------------------------------------------------------------
# Inline training loop (captures metrics + supports Optuna pruning)
# ---------------------------------------------------------------------------

def _run_and_capture(args, results: dict, trial: optuna.Trial):
    """
    Runs the fine-tuning loop inline (mirrors train.py's train() logic) so we
    can:
      1. Report intermediate val_loss to Optuna for pruning.
      2. Capture final metrics without modifying train.py.
    """
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    from dataset_v2 import SpaceflightDataset, make_dataloaders
    from losses import CVAELoss, KLAnnealer
    from model import SpaceflightCVAE
    from train import run_epoch   # reuse existing epoch runner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    pre_target_cats = None
    if args.pretrain_checkpoint and os.path.exists(args.pretrain_checkpoint):
        try:
            ckpt_meta = torch.load(args.pretrain_checkpoint, map_location="cpu")
            pre_target_cats = ckpt_meta.get("target_metadata_categories")
        except Exception:
            pass

    dataset = SpaceflightDataset(args.data, target_metadata_categories=pre_target_cats)
    train_loader, val_loader, _ = make_dataloaders(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # --- Model ---
    hidden_dims = getattr(args, "hidden_dims", [256, 128])
    model = SpaceflightCVAE(
        n_genes=dataset.n_genes,
        n_strains=dataset.n_strains,
        n_sexes=dataset.n_sexes,
        n_tissues=dataset.n_tissues,
        n_euths=dataset.n_euths,
        n_studies=dataset.n_studies,
        conditions=args.conditions,
        latent_dim=args.latent_dim,
        tissue_emb_dim=args.tissue_emb_dim,
        strain_emb_dim=args.strain_emb_dim,
        sex_emb_dim=args.sex_emb_dim,
        study_emb_dim=args.study_emb_dim,
        euth_emb_dim=args.euth_emb_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        grl_alpha=args.grl_alpha,
    ).to(device)

    # --- Weight transfer ---
    if args.pretrain_checkpoint and os.path.exists(args.pretrain_checkpoint):
        ckpt = torch.load(args.pretrain_checkpoint, map_location=device,
                          weights_only=False)
        pretrain_state = ckpt["model_state"]
        model_state    = model.state_dict()
        new_state      = model_state.copy()
        for name, param in pretrain_state.items():
            if name in model_state and param.shape == model_state[name].shape:
                new_state[name] = param.to(device)
        model.load_state_dict(new_state)

    # Freeze decoder if requested
    if getattr(args, "freeze_decoder_epochs", 0) > 0:
        for param in model.decoder.parameters():
            param.requires_grad = False

    # --- Optimizer ---
    if args.pretrain_checkpoint:
        pretrained_params = (list(model.encoder.parameters()) +
                             list(model.decoder.parameters()))
        new_params = (list(model.embedder.parameters()) +
                      list(model.cls_head.parameters()) +
                      list(model.batch_disc.parameters()))
        optimizer = AdamW([
            {"params": pretrained_params, "lr": args.lr},
            {"params": new_params,        "lr": args.lr * args.new_lr_mult},
        ], weight_decay=1e-4)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CVAELoss(beta=args.beta, lambda_cls=args.lambda_cls,
                         lambda_adv=args.lambda_adv)
    annealer  = KLAnnealer(beta=args.beta, n_epochs=args.kl_anneal_epochs)

    # --- Loop ---
    best_val_loss  = float("inf")
    best_val_auroc = 0.5
    last_study_acc = 1.0
    patience_counter = 0
    ckpt_path = Path(args.output_dir) / "best_model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        kl_w = annealer.get(epoch)

        # Unfreeze decoder after freeze period
        freeze_epochs = getattr(args, "freeze_decoder_epochs", 0)
        if (args.pretrain_checkpoint and freeze_epochs > 0
                and epoch == freeze_epochs + 1):
            for param in model.decoder.parameters():
                param.requires_grad = True
            pretrained_params = (list(model.encoder.parameters()) +
                                 list(model.decoder.parameters()))
            new_params = (list(model.embedder.parameters()) +
                          list(model.cls_head.parameters()) +
                          list(model.batch_disc.parameters()))
            optimizer = AdamW([
                {"params": pretrained_params, "lr": args.lr},
                {"params": new_params,        "lr": args.lr * args.new_lr_mult},
            ], weight_decay=1e-4)

        _, _, _ = run_epoch(model, train_loader, criterion, optimizer,
                            device, kl_w, training=True)
        val_metrics, val_acc, val_auroc = run_epoch(
            model, val_loader, criterion, None, device, kl_w, training=False)
        scheduler.step()

        val_loss = val_metrics["loss"]

        # Track best AUROC
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc

        # Study predictability probe every 20 epochs
        if epoch % 20 == 0:
            model.eval()
            zs, studies = [], []
            with torch.no_grad():
                for batch in val_loader:
                    z = model.encode(
                        batch["x"].to(device),
                        batch["strain"].to(device),
                        batch["sex"].to(device),
                        batch["study"].to(device),
                        batch["tissue"].to(device),
                        batch["euth"].to(device),
                        batch["flight"].to(device),
                    )
                    zs.append(z.cpu().numpy())
                    studies.append(batch["study"].numpy())
            zs      = np.concatenate(zs)
            studies = np.concatenate(studies)
            scaler  = StandardScaler()
            clf     = LogisticRegression(max_iter=500, C=1.0, solver="saga")
            clf.fit(scaler.fit_transform(zs), studies)
            last_study_acc = clf.score(scaler.transform(zs), studies)

        # Report to Optuna for pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Checkpointing + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
                "label_encoders": {
                    "strain": list(getattr(dataset.strain_enc, "classes_", [])),
                    "sex":    list(getattr(dataset.sex_enc,    "classes_", [])),
                    "study":  list(getattr(dataset.study_enc,  "classes_", [])),
                    "tissue": list(getattr(dataset.tissue_enc, "classes_", [])),
                    "euth":   list(getattr(dataset.euth_enc,   "classes_", [])),
                },
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    results["best_val_loss"]   = best_val_loss
    results["best_val_auroc"]  = best_val_auroc
    results["last_study_acc"]  = last_study_acc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(study: optuna.Study):
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*60)
    best = study.best_trial
    print(f"Best trial:  #{best.number}")
    print(f"Best score:  {best.value:.4f}")
    print(f"  val_loss:  {best.user_attrs.get('val_loss',  'n/a'):.4f}")
    print(f"  study_acc: {best.user_attrs.get('study_acc', 'n/a'):.4f}")
    print(f"  val_auroc: {best.user_attrs.get('val_auroc', 'n/a'):.4f}")
    print("\nBest hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Top 5 trials
    trials = sorted([t for t in study.trials
                     if t.value is not None], key=lambda t: t.value)
    print("\nTop 5 trials:")
    print(f"  {'Trial':>6}  {'Score':>8}  {'val_loss':>9}  "
          f"{'study_acc':>10}  {'auroc':>7}  conditions  latent_dim")
    for t in trials[:5]:
        print(f"  {t.number:>6}  {t.value:>8.4f}  "
              f"{t.user_attrs.get('val_loss',0):>9.4f}  "
              f"{t.user_attrs.get('study_acc',0):>10.4f}  "
              f"{t.user_attrs.get('val_auroc',0):>7.4f}  "
              f"{t.user_attrs.get('conditions','?')}  "
              f"{t.user_attrs.get('latent_dim','?')}")

    # Suggested CLI for best trial
    p = best.params
    conditions = []
    for c in ["tissue", "strain", "sex", "euth"]:
        if p.get(f"use_{c}", True):
            conditions.append(c)
    cond_str = " ".join(conditions) if conditions else "tissue"

    print("\nRun best trial with train.py:")
    print(f"  python train.py \\")
    print(f"    --latent_dim {p.get('latent_dim', 64)} \\")
    print(f"    --conditions {cond_str} \\")
    for c in ["tissue", "strain", "sex", "euth"]:
        dim = p.get(f"{c}_emb_dim")
        if dim:
            print(f"    --{c}_emb_dim {dim} \\")
    print(f"    --lambda_adv {p.get('lambda_adv', 1.0):.3f} \\")
    print(f"    --lambda_cls {p.get('lambda_cls', 1.0):.3f} \\")
    print(f"    --beta {p.get('beta', 1.0):.3f} \\")
    print(f"    --lr {p.get('lr', 5e-5):.2e} \\")
    print(f"    --dropout {p.get('dropout', 0.2):.2f} \\")
    print(f"    --kl_anneal_epochs {p.get('kl_anneal_epochs', 100)} \\")
    print(f"    --freeze_decoder_epochs {p.get('freeze_decoder_epochs', 0)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HP search for SpaceflightCVAE")

    # Required
    parser.add_argument("--data",         type=str, required=True,
                        help="Path to spaceflight H5 file")
    parser.add_argument("--pretrain_dir", type=str, default=None,
                        help="Directory containing pretrain checkpoints "
                             "(subdirs named <latent_dim>_*/pretrain_best.pt)")

    # Search config
    parser.add_argument("--stage",    type=str, default="arch",
                        choices=["arch", "loss", "train"],
                        help="Search stage: arch → loss → train")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of trials to run (default 50)")
    parser.add_argument("--storage",  type=str, default=None,
                        help="Optuna storage URL e.g. sqlite:///hpsearch.db "
                             "(enables parallel SLURM workers)")
    parser.add_argument("--study_name", type=str, default="spaceflight-cvae",
                        help="Optuna study name (used with --storage)")

    # Fixed architecture overrides for loss/train stages
    parser.add_argument("--latent_dim",   type=int,   default=64)
    parser.add_argument("--hidden_dims",  type=int,   nargs="+", default=[256, 128])
    parser.add_argument("--conditions",   type=str,   nargs="+",
                        default=["tissue"],
                        choices=["tissue", "strain", "sex", "euth"])

    # Fixed training defaults (overridable per stage)
    parser.add_argument("--epochs",           type=int,   default=150)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=5e-5)
    parser.add_argument("--new_lr_mult",      type=float, default=10.0)
    parser.add_argument("--patience",         type=int,   default=30)
    parser.add_argument("--dropout",          type=float, default=0.2)
    parser.add_argument("--grl_alpha",        type=float, default=1.0)
    parser.add_argument("--beta",             type=float, default=1.0)
    parser.add_argument("--lambda_cls",       type=float, default=1.0)
    parser.add_argument("--lambda_adv",       type=float, default=1.0)
    parser.add_argument("--kl_anneal_epochs", type=int,   default=100)
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--freeze_decoder_epochs", type=int, default=0)

    # Condition embedding dims (fixed defaults, searched in arch stage)
    parser.add_argument("--tissue_emb_dim", type=int, default=16)
    parser.add_argument("--strain_emb_dim", type=int, default=8)
    parser.add_argument("--sex_emb_dim",    type=int, default=4)
    parser.add_argument("--study_emb_dim",  type=int, default=8)
    parser.add_argument("--euth_emb_dim",   type=int, default=4)

    # Output
    parser.add_argument("--output_dir",         type=str, default="./hpsearch_checkpoints")
    parser.add_argument("--pretrain_checkpoint", type=str, default=None,
                        help="Fixed pretrain checkpoint (overrides --pretrain_dir)")
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    # Build Optuna study
    sampler = TPESampler(seed=42)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=20)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
    )

    print(f"Optuna search  stage={args.stage}  n_trials={args.n_trials}")
    if args.storage:
        print(f"Storage: {args.storage}  (parallel workers OK)")
    print()

    objective = make_objective(args, args.stage)

    # Optional wandb callback for the study-level dashboard
    callbacks = []
    if WANDB_AVAILABLE and not args.no_wandb:
        try:
            from optuna.integration.wandb import WeightsAndBiasesCallback
            callbacks.append(WeightsAndBiasesCallback(
                metric_name="composite_score",
                wandb_kwargs={"project": "spaceflight-cvae-hpsearch",
                              "name": f"{args.study_name}_{args.stage}"},
            ))
        except ImportError:
            print("optuna wandb integration not available; skipping.")

    study.optimize(objective, n_trials=args.n_trials, callbacks=callbacks)

    print_summary(study)
