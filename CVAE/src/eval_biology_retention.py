"""
Biology Retention Diagnostic for Spaceflight cVAE
====================================================
Answers: "How much general biological structure did fine-tuning on the
2,000-sample flight/ground task cost, relative to the pretrained model?"

Two comparisons, both run on a held-out PROBE dataset (e.g. a slice of
your 500k pretrain samples, or any GeneLab/ARCHS4-style H5 not used in
fine-tuning):

  1. RECONSTRUCTION FIDELITY
     NB reconstruction loss on the probe set, pretrained vs fine-tuned.
     Rising loss after fine-tuning = the encoder/decoder pair is
     explaining general expression patterns worse than before.

  2. TISSUE STRUCTURE IN z
     A linear probe (z -> tissue) fit on each model's latent space.
     Falling tissue-probe accuracy after fine-tuning = biology unrelated
     to flight/ground is being crowded out of the latent space.

A NOTE ON CONDITION VOCAB
--------------------------
Your probe dataset almost certainly has a different categorical vocab
(different studies, possibly different tissue label sets) than what the
checkpoints were originally trained on. Silently re-encoding probe labels
with fresh LabelEncoders and feeding them through the checkpoints would
misalign embedding indices without any error being raised — a wrong
number that looks like a right number.

To avoid that, this script builds BOTH the "baseline" (pretrained) and
"finetuned" models as fresh skeletons conditioned on tissue ONLY, sized
to the PROBE dataset's own vocab, with the SAME random seed before each
build. transfer_pretrained_weights() (imported from train.py) then loads
whatever weights match by name and shape:
  - encoder.net / decoder.net LATER layers, and encoder.mu / encoder.logvar
    (which don't depend on condition dim) transfer with full fidelity.
  - The very FIRST encoder/decoder layer mixes gene + condition dims
    together, so it depends on cond_dim, which will generally mismatch
    the probe skeleton's — it gets skipped and stays at its random init.
    Because both models share the same seed, this random init is
    IDENTICAL across baseline and finetuned, so it cancels out as a
    confound rather than biasing the comparison.
  - Tissue embedding itself is treated the same way: skipped (random,
    seed-matched) unless the probe's tissue vocab size happens to match
    the checkpoint's exactly.

Net effect: this isolates the encoder/decoder FEATURE EXTRACTION pathway
and the mu/logvar latent projection — the part most relevant to "did we
forget general biology" — while sidestepping vocab-mismatch confounds.

Usage:
    python eval_biology_retention.py \\
        --probe_data held_out_pretrain_slice.h5 \\
        --pretrain_checkpoint pretrain_best.pt \\
        --finetuned_checkpoint checkpoints/best_model.pt \\
        --latent_dim 32 --hidden_dims 512 256
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import SpaceflightDataset
from model import SpaceflightCVAE
from losses import nb_nll_loss
from train import transfer_pretrained_weights


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_probe_skeleton(dataset, latent_dim, hidden_dims, dropout, seed, device):
    """
    Build a fresh SpaceflightCVAE conditioned on tissue only, sized to the
    probe dataset's own vocab. Seeded so baseline and finetuned skeletons
    get IDENTICAL random initialization before weight transfer.
    """
    torch.manual_seed(seed)
    model = SpaceflightCVAE(
        n_genes=dataset.n_genes,
        n_strains=dataset.n_strains,
        n_sexes=dataset.n_sexes,
        n_studies=dataset.n_studies,
        n_tissues=dataset.n_tissues,
        n_euths=dataset.n_euths,
        conditions=["tissue"],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        grl_alpha=0.0,
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# Evaluation passes
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_and_reconstruct(model, loader, device):
    """
    Deterministic pass (uses mu, not a sampled z) over the probe set.
    Returns: mean NB reconstruction loss, mu array (N, latent_dim),
    tissue_ids array (N,).
    """
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_mu      = []
    all_tissue  = []

    for batch in loader:
        x       = batch["x"].to(device)
        x_raw   = batch["x_raw"].to(device)
        tissue  = batch["tissue"].to(device)
        zeros   = torch.zeros_like(tissue)

        mu = model.encode(x, zeros, zeros, zeros, tissue, zeros, zeros)
        log_r, p = model.generate(mu, zeros, zeros, zeros, tissue, zeros, zeros)

        loss = nb_nll_loss(x_raw, log_r, p)
        total_loss += loss.item()
        n_batches  += 1

        all_mu.append(mu.cpu().numpy())
        all_tissue.append(tissue.cpu().numpy())

    mean_loss = total_loss / n_batches
    mu_arr     = np.concatenate(all_mu)
    tissue_arr = np.concatenate(all_tissue)
    return mean_loss, mu_arr, tissue_arr


def tissue_probe_accuracy(mu, tissue_ids, test_size=0.2, seed=42):
    """Linear probe: how well does z predict tissue identity?"""
    if len(np.unique(tissue_ids)) < 2:
        return float("nan"), float("nan")

    mu_train, mu_test, y_train, y_test = train_test_split(
        mu, tissue_ids, test_size=test_size, random_state=seed,
        stratify=tissue_ids if len(np.unique(tissue_ids)) <= len(tissue_ids) // 5 else None,
    )
    scaler = StandardScaler()
    mu_train_s = scaler.fit_transform(mu_train)
    mu_test_s  = scaler.transform(mu_test)

    clf = LogisticRegression(max_iter=500, C=0.1)
    clf.fit(mu_train_s, y_train)

    train_acc = clf.score(mu_train_s, y_train)
    test_acc  = clf.score(mu_test_s, y_test)
    return train_acc, test_acc


def weight_drift_by_layer(pretrain_ckpt_path, finetuned_ckpt_path, device):
    """
    Layer-by-layer relative L2 drift between the pretrained and fine-tuned
    encoder/decoder body weights, using the same key-remapping table as
    transfer_pretrained_weights(). No probe data needed — this is a pure
    checkpoint-to-checkpoint comparison.

    Large drift in EARLY encoder layers (generic feature extraction) is
    the concerning signal; drift near the latent bottleneck (mu/logvar)
    or late decoder layers is more expected, since fine-tuning is
    supposed to adapt those.
    """
    pre_ckpt = torch.load(pretrain_ckpt_path, map_location=device, weights_only=False)
    fin_ckpt = torch.load(finetuned_ckpt_path, map_location=device, weights_only=False)

    pre_state = pre_ckpt["model_state"]
    fin_state = fin_ckpt["model_state"]

    key_map = {
        "encoder_net.":  "encoder.net.",
        "mu_head.":      "encoder.mu.",
        "logvar_head.":  "encoder.logvar.",
        "decoder_net.":  "decoder.net.",
        "log_r_head.":   "decoder.log_r_head.",
        "p_head.":       "decoder.p_head.",
    }

    results = []
    for pre_key, pre_val in pre_state.items():
        if any(pre_key.startswith(p) for p in ["cls_head.", "batch_disc."]):
            continue
        if pre_key.startswith("embedder."):
            continue
        if not pre_key.endswith(".weight"):
            continue  # skip biases, one number per layer is enough signal

        mapped_key = pre_key
        for src, dst in key_map.items():
            if pre_key.startswith(src):
                mapped_key = pre_key.replace(src, dst)
                break

        if mapped_key not in fin_state or fin_state[mapped_key].shape != pre_val.shape:
            continue

        fin_val = fin_state[mapped_key]
        drift = (fin_val - pre_val).norm().item() / (pre_val.norm().item() + 1e-8)
        results.append((mapped_key, drift))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print("Loading probe dataset...")
    dataset = SpaceflightDataset(args.probe_data)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print("\n" + "=" * 60)
    print("Building baseline (pretrained) model")
    print("=" * 60)
    baseline_model = build_probe_skeleton(
        dataset, args.latent_dim, args.hidden_dims, dropout=0.0,
        seed=args.seed, device=device,
    )
    transfer_pretrained_weights(baseline_model, args.pretrain_checkpoint, device)

    print("\n" + "=" * 60)
    print("Building fine-tuned model")
    print("=" * 60)
    finetuned_model = build_probe_skeleton(
        dataset, args.latent_dim, args.hidden_dims, dropout=0.0,
        seed=args.seed, device=device,
    )
    transfer_pretrained_weights(finetuned_model, args.finetuned_checkpoint, device)

    print("\n" + "=" * 60)
    print("Running reconstruction + tissue-probe evaluation")
    print("=" * 60)

    base_loss, base_mu, tissue_ids = encode_and_reconstruct(baseline_model, loader, device)
    fin_loss,  fin_mu,  _          = encode_and_reconstruct(finetuned_model, loader, device)

    base_train_acc, base_test_acc = tissue_probe_accuracy(base_mu, tissue_ids, seed=args.seed)
    fin_train_acc,  fin_test_acc  = tissue_probe_accuracy(fin_mu, tissue_ids, seed=args.seed)

    print(f"\n{'Metric':<32}{'Pretrained':>15}{'Fine-tuned':>15}{'Delta':>12}")
    print("-" * 74)
    print(f"{'NB reconstruction loss':<32}{base_loss:>15.4f}{fin_loss:>15.4f}"
          f"{fin_loss - base_loss:>+12.4f}")
    print(f"{'Tissue probe (train acc)':<32}{base_train_acc:>15.3f}{fin_train_acc:>15.3f}"
          f"{fin_train_acc - base_train_acc:>+12.3f}")
    print(f"{'Tissue probe (test acc)':<32}{base_test_acc:>15.3f}{fin_test_acc:>15.3f}"
          f"{fin_test_acc - base_test_acc:>+12.3f}")

    if fin_loss > base_loss * 1.1:
        print("\n\u26a0 Reconstruction loss rose >10% after fine-tuning — "
              "possible forgetting of general biology.")
    if fin_test_acc < base_test_acc - 0.05:
        print("\u26a0 Tissue probe test accuracy dropped >5 points after "
              "fine-tuning — biology unrelated to flight/ground may be "
              "getting crowded out of z.")
    if fin_loss <= base_loss * 1.1 and fin_test_acc >= base_test_acc - 0.05:
        print("\nNo strong evidence of forgetting on these two signals.")

    print("\n" + "=" * 60)
    print("Per-layer weight drift (pretrained -> fine-tuned)")
    print("=" * 60)
    drift_results = weight_drift_by_layer(
        args.pretrain_checkpoint, args.finetuned_checkpoint, device
    )
    if drift_results:
        for key, drift in drift_results:
            flag = "  <-- large drift" if drift > 0.5 else ""
            print(f"  {key:<40} relative L2 drift = {drift:.3f}{flag}")
        print("\n  (Drift concentrated in mu/logvar or late decoder layers "
              "is expected. Large drift in early encoder.net layers is "
              "the more concerning pattern — that's generic feature "
              "extraction, not classification-specific adaptation.)")
    else:
        print("  No matching weight keys found between checkpoints — "
              "check that both checkpoints share the same architecture.")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Biology retention diagnostic")
    parser.add_argument("--probe_data",           type=str, required=True,
                        help="H5 file with held-out biology probe data "
                             "(e.g. a slice of pretrain data, not used in fine-tuning)")
    parser.add_argument("--pretrain_checkpoint",   type=str, required=True)
    parser.add_argument("--finetuned_checkpoint",  type=str, required=True)
    parser.add_argument("--latent_dim",  type=int, default=32,
                        help="Must match the architecture used for both checkpoints")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256],
                        help="Must match the architecture used for both checkpoints")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed",        type=int, default=42,
                        help="Shared seed for baseline/finetuned skeleton init "
                             "so untransferred layers match exactly")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)
