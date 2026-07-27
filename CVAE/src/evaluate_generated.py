"""
Evaluate Generated / Counterfactual Samples
=============================================
Measures the quality of samples produced by generate.py or whatif.py
against real samples from subset_final.h5.

Five evaluation metrics:
  1. Statistical fidelity     — mean/variance correlation between real and generated
  2. Library size distribution — KS test on per-sample total counts
  3. Latent indistinguishability — can a classifier tell real from generated in z?
  4. Biological coherence     — correlation of LFC between whatif and real DE
  5. Fréchet distance         — distribution distance in latent space

Usage:

    # evaluate unconditional synthetic samples (from generate.py)
    python evaluate_generated.py \\
        --checkpoint checkpoints/finetune/v2/best_model.pt \\
        --data subset_final.h5 \\
        --generated synthetic_samples/Liver_C57BL-6J_Female_1_n100/synthetic_expression.csv \\
        --mode synthetic \\
        --tissue Liver --strain C57BL/6J --sex Female --flight 1

    # evaluate counterfactual samples (from whatif.py population mode)
    python evaluate_generated.py \\
        --checkpoint checkpoints/finetune/v2/best_model.pt \\
        --data subset_final.h5 \\
        --generated whatif_results/population_flight_expression.csv \\
        --real_de de_results/liver_de.csv \\
        --mode counterfactual \\
        --tissue Liver --flight 1

Outputs (saved to output_dir/):
    evaluation_summary.txt      all metrics in one report
    metric1_mean_var.csv        per-gene mean/variance comparison
    metric3_latent_roc.png      ROC curve for real vs generated classifier
    metric4_lfc_scatter.png     whatif LFC vs real LFC scatter (counterfactual only)
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, ks_2samp
from scipy.linalg import sqrtm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import SpaceflightDataset
from model import SpaceflightCVAE


# ---------------------------------------------------------------------------
# Model loading and encoding
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, dataset, device):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args  = ckpt["args"]
    model = SpaceflightCVAE(
        n_genes=dataset.n_genes,
        n_strains=dataset.n_strains,
        n_sexes=dataset.n_sexes,
        n_studies=dataset.n_studies,
        n_tissues=dataset.n_tissues,
        n_euths=dataset.n_euths,
        latent_dim=args["latent_dim"],
        hidden_dims=[256, 128],
        dropout=0.0,
        grl_alpha=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded model (epoch {ckpt['epoch']}, latent_dim={args['latent_dim']})")
    return model, args["latent_dim"]


def encode_expression(model, dataset, expr_matrix, condition_kwargs, device,
                      batch_size=256):
    """
    Encode an arbitrary expression matrix (n_samples, n_genes) to latent space.
    condition_kwargs: dict with keys strain/sex/study/tissue/euth/flight
                      each can be an int (same for all) or array (per sample)
    """
    n = expr_matrix.shape[0]
    all_z = []

    def make_tensor(val, n):
        if np.isscalar(val):
            return torch.full((n,), val, dtype=torch.long)
        return torch.tensor(val, dtype=torch.long)

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = expr_matrix[start:end]

        # normalize: log1p(CPM)
        lib  = batch.sum(axis=1, keepdims=True)
        lib  = np.maximum(lib, 1.0)
        x    = np.log1p(batch / lib * 1e4).astype(np.float32)

        with torch.no_grad():
            mu = model.encode(
                torch.tensor(x).to(device),
                make_tensor(condition_kwargs["strain"], end-start).to(device),
                make_tensor(condition_kwargs["sex"],    end-start).to(device),
                make_tensor(condition_kwargs["study"],  end-start).to(device),
                make_tensor(condition_kwargs["tissue"], end-start).to(device),
                make_tensor(condition_kwargs["euth"],   end-start).to(device),
                make_tensor(condition_kwargs["flight"], end-start).to(device),
            )
        all_z.append(mu.cpu().numpy())

    return np.concatenate(all_z, axis=0)


def get_real_samples(dataset, tissue=None, strain=None, sex=None,
                     flight=None, euth=None):
    """Select real samples matching metadata filters."""
    mask = np.ones(dataset.n_samples, dtype=bool)
    if tissue  is not None:
        t = dataset.tissue_enc.transform([tissue])[0]
        mask &= dataset.tissue_ids == t
    if strain  is not None:
        s = dataset.strain_enc.transform([strain])[0]
        mask &= dataset.strain_ids == s
    if sex     is not None:
        sx = dataset.sex_enc.transform([sex])[0]
        mask &= dataset.sex_ids == sx
    if flight  is not None:
        mask &= dataset.flight == flight
    if euth    is not None:
        e = dataset.euth_enc.transform([euth])[0]
        mask &= dataset.euth_ids == e

    indices = np.where(mask)[0]
    expr    = dataset.raw_counts[indices].astype(np.float32)
    return indices, expr


# ---------------------------------------------------------------------------
# Metric 1: Statistical fidelity
# ---------------------------------------------------------------------------

def metric_statistical_fidelity(real_expr, gen_expr, gene_symbols, out_dir):
    """Mean and variance correlation between real and generated."""
    print("\n--- Metric 1: Statistical Fidelity ---")

    # normalize both to log1p CPM for comparison
    def to_logcpm(expr):
        lib = np.maximum(expr.sum(axis=1, keepdims=True), 1.0)
        return np.log1p(expr / lib * 1e4)

    real_norm = to_logcpm(real_expr)
    gen_norm  = to_logcpm(gen_expr)

    real_mean = real_norm.mean(axis=0)
    gen_mean  = gen_norm.mean(axis=0)
    real_var  = real_norm.var(axis=0)
    gen_var   = gen_norm.var(axis=0)

    r_mean, p_mean   = pearsonr(real_mean, gen_mean)
    rho_mean, _      = spearmanr(real_mean, gen_mean)
    r_var,  p_var    = pearsonr(real_var,  gen_var)
    rho_var, _       = spearmanr(real_var,  gen_var)

    # fraction of zeros
    real_zeros = (real_expr == 0).mean()
    gen_zeros  = (gen_expr  == 0).mean()

    print(f"  Mean expression  Pearson r={r_mean:.4f}  Spearman ρ={rho_mean:.4f}")
    print(f"  Variance         Pearson r={r_var:.4f}   Spearman ρ={rho_var:.4f}")
    print(f"  Zero fraction    real={real_zeros:.3f}   generated={gen_zeros:.3f}")

    # save per-gene comparison
    df = pd.DataFrame({
        "symbol":    gene_symbols,
        "real_mean": real_mean,
        "gen_mean":  gen_mean,
        "real_var":  real_var,
        "gen_var":   gen_var,
        "mean_diff": gen_mean - real_mean,
    }).sort_values("mean_diff", key=abs, ascending=False)
    df.to_csv(out_dir / "metric1_mean_var.csv", index=False)

    return {
        "mean_pearson_r":    r_mean,
        "mean_pearson_p":    p_mean,
        "mean_spearman_rho": rho_mean,
        "var_pearson_r":     r_var,
        "var_pearson_p":     p_var,
        "var_spearman_rho":  rho_var,
        "real_zero_fraction":real_zeros,
        "gen_zero_fraction": gen_zeros,
    }


# ---------------------------------------------------------------------------
# Metric 2: Library size distribution
# ---------------------------------------------------------------------------

def metric_library_size(real_expr, gen_expr):
    """KS test on library size distributions."""
    print("\n--- Metric 2: Library Size Distribution ---")

    real_lib = real_expr.sum(axis=1)
    gen_lib  = gen_expr.sum(axis=1)

    ks_stat, ks_p = ks_2samp(real_lib, gen_lib)

    print(f"  Real  library size: mean={real_lib.mean():,.0f}  "
          f"median={np.median(real_lib):,.0f}  std={real_lib.std():,.0f}")
    print(f"  Gen   library size: mean={gen_lib.mean():,.0f}  "
          f"median={np.median(gen_lib):,.0f}  std={gen_lib.std():,.0f}")
    print(f"  KS test: stat={ks_stat:.4f}  p={ks_p:.4f}  "
          f"({'indistinguishable' if ks_p > 0.05 else 'significantly different'})")

    return {
        "real_lib_mean":   float(real_lib.mean()),
        "gen_lib_mean":    float(gen_lib.mean()),
        "real_lib_median": float(np.median(real_lib)),
        "gen_lib_median":  float(np.median(gen_lib)),
        "ks_stat":         float(ks_stat),
        "ks_pval":         float(ks_p),
    }


# ---------------------------------------------------------------------------
# Metric 3: Latent indistinguishability
# ---------------------------------------------------------------------------

def metric_latent_indistinguishability(z_real, z_gen, out_dir):
    """
    Can a classifier distinguish real from generated in latent space?
    AUROC close to 0.5 = indistinguishable (good).
    AUROC close to 1.0 = easily distinguishable (bad).
    """
    print("\n--- Metric 3: Latent Indistinguishability ---")

    X = np.vstack([z_real, z_gen])
    y = np.array([0] * len(z_real) + [1] * len(z_gen))

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        random_state=42, n_jobs=-1
    )
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs  = cross_val_predict(clf, X_sc, y, cv=cv, method="predict_proba")[:, 1]
    auroc  = roc_auc_score(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)

    print(f"  Real vs generated AUROC (latent space): {auroc:.4f}")
    print(f"  Interpretation: "
          f"{'excellent (near random)' if auroc < 0.6 else 'good' if auroc < 0.7 else 'moderate' if auroc < 0.8 else 'poor — distributions differ substantially'}")

    # ROC curve plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Real vs Generated — Latent Space Classifier")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "metric3_latent_roc.png", dpi=150)
    plt.close()

    return {"latent_auroc": float(auroc)}


# ---------------------------------------------------------------------------
# Metric 4: Biological coherence (LFC correlation)
# ---------------------------------------------------------------------------

def metric_biological_coherence(real_de_path, gen_expr, real_expr,
                                 gene_symbols, out_dir):
    """
    Correlation between model-predicted LFC (whatif/generate) and
    observed DE LFC from real data.
    Only meaningful for counterfactual mode.
    """
    print("\n--- Metric 4: Biological Coherence ---")

    if real_de_path is None:
        print("  Skipped (--real_de not provided)")
        return {}

    de_df = pd.read_csv(real_de_path)
    if "lfc" not in de_df.columns or "symbol" not in de_df.columns:
        print("  Skipped — DE file must have 'symbol' and 'lfc' columns")
        return {}

    # normalize both
    def to_logcpm(expr):
        lib = np.maximum(expr.sum(axis=1, keepdims=True), 1.0)
        return np.log1p(expr / lib * 1e4)

    gen_norm  = to_logcpm(gen_expr)
    real_norm = to_logcpm(real_expr)

    gen_mean  = gen_norm.mean(axis=0)
    real_mean = real_norm.mean(axis=0)

    # model-predicted LFC = generated - real (for counterfactual flight vs ground)
    predicted_lfc = gen_mean - real_mean

    # match against real DE
    sym_to_pred = dict(zip(gene_symbols, predicted_lfc))
    de_df["predicted_lfc"] = de_df["symbol"].map(sym_to_pred)
    de_df = de_df.dropna(subset=["predicted_lfc"])

    r, p     = pearsonr(de_df["lfc"], de_df["predicted_lfc"])
    rho, rho_p = spearmanr(de_df["lfc"], de_df["predicted_lfc"])

    # sign concordance
    concordant = (np.sign(de_df["lfc"]) == np.sign(de_df["predicted_lfc"])).mean()

    print(f"  LFC correlation: Pearson r={r:.4f} (p={p:.2e})")
    print(f"                   Spearman ρ={rho:.4f} (p={rho_p:.2e})")
    print(f"  Sign concordance: {concordant:.1%} of genes in same direction")

    # scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(de_df["lfc"], de_df["predicted_lfc"],
               alpha=0.3, s=8, color="steelblue", edgecolors="none")
    lim = max(abs(de_df["lfc"].max()), abs(de_df["predicted_lfc"].max())) * 1.1
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Observed LFC (real DE)")
    ax.set_ylabel("Predicted LFC (model)")
    ax.set_title(f"Biological Coherence\nPearson r={r:.3f}  Spearman ρ={rho:.3f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "metric4_lfc_scatter.png", dpi=150)
    plt.close()

    de_df.to_csv(out_dir / "metric4_lfc_comparison.csv", index=False)

    return {
        "lfc_pearson_r":     float(r),
        "lfc_pearson_p":     float(p),
        "lfc_spearman_rho":  float(rho),
        "lfc_spearman_p":    float(rho_p),
        "sign_concordance":  float(concordant),
    }


# ---------------------------------------------------------------------------
# Metric 5: Fréchet distance in latent space
# ---------------------------------------------------------------------------

def metric_frechet_distance(z_real, z_gen):
    """
    Fréchet distance between real and generated latent distributions.
    Lower is better; 0 = identical distributions.
    """
    print("\n--- Metric 5: Fréchet Distance (Latent Space) ---")

    mu1    = z_real.mean(axis=0)
    mu2    = z_gen.mean(axis=0)
    sigma1 = np.cov(z_real.T) if len(z_real) > 1 else np.eye(z_real.shape[1])
    sigma2 = np.cov(z_gen.T)  if len(z_gen)  > 1 else np.eye(z_gen.shape[1])

    diff     = mu1 - mu2
    covmean  = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

    print(f"  Fréchet distance (latent): {fd:.4f}")
    print(f"  Mean distance (centroids): {np.linalg.norm(diff):.4f}")
    print(f"  Interpretation: "
          f"{'excellent' if fd < 10 else 'good' if fd < 50 else 'moderate' if fd < 200 else 'poor'}")

    return {
        "frechet_distance": fd,
        "centroid_distance": float(np.linalg.norm(diff)),
    }


# ---------------------------------------------------------------------------
# Write summary report
# ---------------------------------------------------------------------------

def write_summary(metrics, args, n_real, n_gen, out_dir):
    lines = [
        "=== Generated Sample Evaluation Summary ===\n",
        f"Mode:           {args.mode}",
        f"Checkpoint:     {args.checkpoint}",
        f"Generated file: {args.generated}",
        f"Real samples:   {n_real}",
        f"Generated:      {n_gen}",
    ]
    if args.tissue:  lines.append(f"Tissue filter:  {args.tissue}")
    if args.strain:  lines.append(f"Strain filter:  {args.strain}")
    if args.sex:     lines.append(f"Sex filter:     {args.sex}")
    if args.flight is not None: lines.append(f"Flight filter:  {args.flight}")
    lines.append("")

    sections = {
        "Metric 1: Statistical Fidelity": [
            ("Mean expression Pearson r",   "mean_pearson_r",    ".4f"),
            ("Mean expression Spearman ρ",  "mean_spearman_rho", ".4f"),
            ("Variance Pearson r",          "var_pearson_r",     ".4f"),
            ("Variance Spearman ρ",         "var_spearman_rho",  ".4f"),
            ("Real zero fraction",          "real_zero_fraction",".3f"),
            ("Generated zero fraction",     "gen_zero_fraction", ".3f"),
        ],
        "Metric 2: Library Size Distribution": [
            ("Real mean library size",   "real_lib_mean",   ",.0f"),
            ("Generated mean lib size",  "gen_lib_mean",    ",.0f"),
            ("KS statistic",             "ks_stat",         ".4f"),
            ("KS p-value",               "ks_pval",         ".4f"),
        ],
        "Metric 3: Latent Indistinguishability": [
            ("Real vs generated AUROC (latent)", "latent_auroc", ".4f"),
        ],
        "Metric 4: Biological Coherence (LFC)": [
            ("LFC Pearson r",        "lfc_pearson_r",    ".4f"),
            ("LFC Spearman ρ",       "lfc_spearman_rho", ".4f"),
            ("Sign concordance",     "sign_concordance", ".1%"),
        ],
        "Metric 5: Fréchet Distance (Latent)": [
            ("Fréchet distance",     "frechet_distance",  ".4f"),
            ("Centroid distance",    "centroid_distance", ".4f"),
        ],
    }

    for section, fields in sections.items():
        lines.append(f"\n{section}:")
        for label, key, fmt in fields:
            val = metrics.get(key)
            if val is None:
                lines.append(f"  {label}: N/A")
            else:
                try:
                    lines.append(f"  {label}: {format(val, fmt)}")
                except (TypeError, ValueError):
                    lines.append(f"  {label}: {val}")

    lines.append("\n\nInterpretation Guide:")
    lines.append("  Mean expression Pearson r > 0.95  → good statistical fidelity")
    lines.append("  Library size KS p > 0.05          → indistinguishable distributions")
    lines.append("  Latent AUROC < 0.60               → generated samples blend with real")
    lines.append("  LFC Pearson r > 0.70              → biologically coherent (counterfactual)")
    lines.append("  Fréchet distance < 50             → close latent distributions")

    report = "\n".join(lines)
    print("\n" + report)

    with open(out_dir / "evaluation_summary.txt", "w") as f:
        f.write(report)
    print(f"\nSaved: {out_dir / 'evaluation_summary.txt'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load dataset and model
    print("\nLoading dataset...")
    dataset = SpaceflightDataset(args.data)

    print("\nLoading model...")
    model, latent_dim = load_model(args.checkpoint, dataset, device)

    # load generated expression
    print(f"\nLoading generated samples from {args.generated}...")
    gen_df   = pd.read_csv(args.generated, index_col=0)
    gen_expr = gen_df.values.astype(np.float32)
    print(f"  Generated: {gen_expr.shape[0]} samples x {gen_expr.shape[1]} genes")

    # check if generated file is already log1p normalized or raw counts
    # (files from generate.py synthetic_counts.csv are raw; expression.csv is normalized)
    if "count" in args.generated.lower():
        print("  Detected raw counts file (synthetic_counts.csv)")
        gen_raw = gen_expr
    else:
        print("  Detected normalized expression file — back-transforming to pseudo-counts")
        # approximate raw counts: expm1(x) * median_library_size / 1e4
        gen_raw = np.expm1(gen_expr) * 1e4

    # get real samples matching filters
    print("\nSelecting real samples...")
    real_indices, real_expr = get_real_samples(
        dataset,
        tissue=args.tissue,
        strain=args.strain,
        sex=args.sex,
        flight=args.flight,
        euth=args.euth,
    )
    print(f"  Real: {len(real_indices)} samples")

    if len(real_indices) < 5:
        raise ValueError(
            f"Only {len(real_indices)} real samples match the filters — need at least 5."
        )

    # resolve condition encodings for latent encoding
    def resolve(val, encoder):
        if val is None: return 0
        return int(encoder.transform([val])[0])

    cond = {
        "tissue": resolve(args.tissue, dataset.tissue_enc),
        "strain": resolve(args.strain, dataset.strain_enc),
        "sex":    resolve(args.sex,    dataset.sex_enc),
        "euth":   resolve(args.euth,   dataset.euth_enc),
        "study":  0,
        "flight": int(args.flight) if args.flight is not None else 0,
    }

    # encode both sets
    print("\nEncoding real samples...")
    z_real = encode_expression(model, dataset,
                               real_expr, cond, device)

    print("Encoding generated samples...")
    z_gen  = encode_expression(model, dataset,
                               gen_raw,   cond, device)

    print(f"  z_real: {z_real.shape}  z_gen: {z_gen.shape}")

    # run all metrics
    all_metrics = {}

    m1 = metric_statistical_fidelity(
        real_expr, gen_raw, dataset.gene_symbols, out_dir
    )
    all_metrics.update(m1)

    m2 = metric_library_size(real_expr, gen_raw)
    all_metrics.update(m2)

    m3 = metric_latent_indistinguishability(z_real, z_gen, out_dir)
    all_metrics.update(m3)

    m4 = metric_biological_coherence(
        args.real_de, gen_raw, real_expr,
        dataset.gene_symbols, out_dir
    )
    all_metrics.update(m4)

    m5 = metric_frechet_distance(z_real, z_gen)
    all_metrics.update(m5)

    # write summary
    write_summary(all_metrics, args,
                  n_real=len(real_indices), n_gen=len(gen_raw),
                  out_dir=out_dir)

    # save raw metrics as CSV
    pd.DataFrame([all_metrics]).to_csv(
        out_dir / "evaluation_metrics.csv", index=False
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated/counterfactual RNA-seq samples"
    )

    # required
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data",        type=str, required=True,
                        help="Path to subset_final.h5")
    parser.add_argument("--generated",   type=str, required=True,
                        help="Path to generated expression CSV "
                             "(from generate.py or whatif.py)")

    # evaluation mode
    parser.add_argument("--mode", type=str, default="synthetic",
                        choices=["synthetic", "counterfactual"],
                        help="synthetic (generate.py) or counterfactual (whatif.py)")

    # real sample filters — used to select matching real samples for comparison
    parser.add_argument("--tissue",  type=str,   default=None)
    parser.add_argument("--strain",  type=str,   default=None)
    parser.add_argument("--sex",     type=str,   default=None)
    parser.add_argument("--flight",  type=int,   default=None, choices=[0, 1])
    parser.add_argument("--euth",    type=str,   default=None)

    # biological coherence (metric 4) — optional, counterfactual mode only
    parser.add_argument("--real_de", type=str,   default=None,
                        help="Path to real DE CSV (from tissue_de_analysis) "
                             "for LFC correlation. Needs 'symbol' and 'lfc' columns.")

    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Output directory")

    args = parser.parse_args()
    run(args)
