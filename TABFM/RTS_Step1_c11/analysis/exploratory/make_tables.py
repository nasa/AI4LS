#!/usr/bin/env python3
"""Render slide-ready tables T1-T5 for the C11 total-hip BMD TabPFN-3 aim.

Reads the verified source CSVs in this folder and writes:
  - reformatted CSVs (T1..T5_*.csv) into this folder
  - rendered table images (PNG 300dpi + SVG, editable text) into figures/

Style matches make_figures.py: Liberation Sans, Okabe-Ito accents, svg.fonttype='none'.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00",
          sky="#56B4E9", grey="#999999", black="#000000", yellow="#F0E442")

HDR = "#2B2B2B"      # header fill
ROW_A = "#FFFFFF"
ROW_B = "#F2F2F2"


def _render_table(df, col_labels, title, out_name, col_widths=None,
                  highlight_rows=None, highlight_color=OI["vermillion"],
                  fontsize=11, title_fs=14, fig_w=11.0, note=None):
    """Render a dataframe as a clean slide table -> PNG + SVG.

    highlight_rows: dict {row_index: color} for left-edge accent + bold.
    """
    n_rows, n_cols = df.shape
    if col_widths is None:
        col_widths = [1.0] * n_cols
    # row heights: header + body (+ optional note line handled by fig height)
    row_h = 0.5
    fig_h = 1.0 + row_h * (n_rows + 1) + (0.5 if note else 0.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(cellText=df.values, colLabels=col_labels,
                     cellLoc="center", colLoc="center",
                     colWidths=[w / sum(col_widths) for w in col_widths],
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.6)

    highlight_rows = highlight_rows or {}
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.6)
        if r == 0:  # header
            cell.set_facecolor(HDR)
            cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor(HDR)
        else:
            body_r = r - 1
            cell.set_facecolor(ROW_B if body_r % 2 else ROW_A)
            if body_r in highlight_rows:
                cell.set_text_props(weight="bold", color=highlight_rows[body_r])
                cell.set_edgecolor(highlight_rows[body_r])
                cell.set_linewidth(1.0)

    ax.set_title(title, fontsize=title_fs, weight="bold", pad=14, loc="left")
    if note:
        fig.text(0.01, 0.01, note, fontsize=fontsize - 2, color="#555555",
                 ha="left", va="bottom")
    fig.tight_layout()
    png = os.path.join(FIG, out_name + ".png")
    svg = os.path.join(FIG, out_name + ".svg")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", png, "and", svg)


def fmt(x, nd=3):
    if pd.isna(x):
        return "—"
    return f"{x:.{nd}f}"


def fmt_p(p):
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# ---------------------------------------------------------------- T1 descriptive
def t1():
    m = pd.read_csv(os.path.join(HERE, "tfm_results_metrics.csv"))
    rows = []
    for _, r in m.iterrows():
        rows.append([
            r["table"].capitalize(),
            f'{r["r2"]:.3f} [{r["r2_ci_lo"]:.3f}, {r["r2_ci_hi"]:.3f}]',
            f'{r["rmse"]:.4f} [{r["rmse_ci_lo"]:.4f}, {r["rmse_ci_hi"]:.4f}]',
            f'{r["mae"]:.4f} [{r["mae_ci_lo"]:.4f}, {r["mae_ci_hi"]:.4f}]',
            f'{r["spearman"]:.3f} [{r["spearman_ci_lo"]:.3f}, {r["spearman_ci_hi"]:.3f}]',
        ])
    # floor row (same for both tables; take from first)
    f = m.iloc[0]
    rows.append([
        "Mean-predictor floor",
        f'{f["floor_r2"]:.3f}',
        f'{f["floor_rmse"]:.4f}',
        f'{f["floor_mae"]:.4f}',
        f'{f["floor_spearman"]:.3f}',
    ])
    df = pd.DataFrame(rows, columns=["Table", "R² [95% CI]", "RMSE [95% CI]",
                                     "MAE [95% CI]", "Spearman [95% CI]"])
    df.to_csv(os.path.join(HERE, "T1_descriptive_metrics.csv"), index=False)
    _render_table(
        df, list(df.columns),
        "T1. Descriptive performance — 10×5 repeated CV (n=38)",
        "T1_descriptive_metrics",
        col_widths=[2.2, 2.4, 2.6, 2.6, 2.6],
        highlight_rows={0: OI["blue"]},
        note=("Descriptive only — never compared to the permutation null. "
              "Everything table (628 features) vs shortlist (5). Floor = predict the training mean."),
        fig_w=12.5, fontsize=10.5,
    )


# ---------------------------------------------------------------- T2 null summary
def t2():
    n = pd.read_csv(os.path.join(HERE, "tfm_results_null.csv"))
    ood = pd.read_csv(os.path.join(HERE, "tfm_results_null_ood.csv"))
    rows = []
    for _, r in n.iterrows():
        rows.append([
            r["table"].capitalize(),
            int(r["n_shuffles"]),
            f'{r["obs_spearman"]:.3f}',
            f'{r["null_spearman_mean"]:.3f} ± {r["null_spearman_sd"]:.3f}',
            f'{r["null_spearman_p95"]:.3f}',
            fmt_p(r["emp_p_spearman"]),
            fmt_p(r["emp_p_neg_mae"]),
        ])
    o = ood.iloc[0]
    rows.append([
        "Everything (OOD ckpt)",
        int(o["n_shuffles"]),
        f'{o["obs_spearman"]:.3f}',
        f'{o["null_spearman_mean"]:.3f} ± {o["null_spearman_sd"]:.3f}',
        f'{o["null_spearman_p95"]:.3f}',
        fmt_p(o["emp_p_spearman"]),
        fmt_p(o["emp_p_neg_mae"]),
    ])
    df = pd.DataFrame(rows, columns=["Table", "Shuffles", "Obs. Spearman",
                                     "Null mean ± SD", "Null p95",
                                     "p (Spearman)", "p (−MAE)"])
    df.to_csv(os.path.join(HERE, "T2_null_summary.csv"), index=False)
    _render_table(
        df, list(df.columns),
        "T2. Permutation-null summary — LOOCV, one-sided empirical p",
        "T2_null_summary",
        col_widths=[2.6, 1.1, 1.6, 2.0, 1.2, 1.5, 1.4],
        highlight_rows={0: OI["blue"], 2: OI["orange"]},
        note=("Null is NOT centered at 0 (LOOCV artifact). Observed 0.376 sits ~0.77 above the null "
              "center (~2.8 null-SD), not 0.38 above zero. OOD = out-of-distribution checkpoint (200 shuffles)."),
        fig_w=12.5, fontsize=10.5,
    )


# ---------------------------------------------------------------- T3 top importance
def t3():
    g = pd.read_csv(os.path.join(HERE, "tfm_group_importance.csv"))
    g = g.sort_values("rank_by_obs_drop_spearman").reset_index(drop=True)
    top = g.head(5).copy()
    bone = g[g["group"] == "Bone_mineral_regional"].copy()
    sel = pd.concat([top, bone]).drop_duplicates("group")
    rows = []
    for _, r in sel.iterrows():
        rows.append([
            r["group"].replace("_", " "),
            int(r["n_features"]),
            int(r["rank_by_obs_drop_spearman"]),
            f'{r["obs_drop_spearman"]:+.3f}',
            f'{r["null_drop_spearman_mean"]:+.3f} ± {r["null_drop_spearman_sd"]:.3f}',
            f'{r["z_spearman"]:+.2f}',
            fmt_p(r["emp_p_spearman"]),
        ])
    df = pd.DataFrame(rows, columns=["Domain group", "Features", "Rank",
                                     "Obs. drop", "Null drop mean ± SD",
                                     "z", "p (Spearman)"])
    df.to_csv(os.path.join(HERE, "T3_top_group_importance.csv"), index=False)
    hl = {0: OI["vermillion"]}
    bone_idx = df.index[df["Domain group"] == "Bone mineral regional"]
    if len(bone_idx):
        hl[int(bone_idx[0])] = OI["sky"]
    _render_table(
        df, list(df.columns),
        "T3. Grouped permutation importance — top 5 + Bone (everything table)",
        "T3_top_group_importance",
        col_widths=[2.6, 1.0, 0.8, 1.2, 2.2, 0.9, 1.3],
        highlight_rows=hl,
        note=("Drop in LOOCV Spearman when the group is permuted. Screening/fitness (CPET) is the largest "
              "contributor; Bone mineral regional does NOT drive the signal (rank 10, z≈−0.06)."),
        fig_w=12.5, fontsize=10.5,
    )


# ---------------------------------------------------------------- T4 full 16 groups
def t4():
    g = pd.read_csv(os.path.join(HERE, "tfm_group_importance.csv"))
    g = g.sort_values("rank_by_obs_drop_spearman").reset_index(drop=True)
    rows = []
    for _, r in g.iterrows():
        rows.append([
            int(r["rank_by_obs_drop_spearman"]),
            r["group"].replace("_", " "),
            int(r["n_features"]),
            f'{r["obs_drop_spearman"]:+.3f}',
            f'{r["null_drop_spearman_mean"]:+.3f} ± {r["null_drop_spearman_sd"]:.3f}',
            f'{r["z_spearman"]:+.2f}',
            fmt_p(r["emp_p_spearman"]),
        ])
    df = pd.DataFrame(rows, columns=["Rank", "Domain group", "Features",
                                     "Obs. drop", "Null drop mean ± SD",
                                     "z", "p (Spearman)"])
    df.to_csv(os.path.join(HERE, "T4_full_group_importance.csv"), index=False)
    hl = {0: OI["vermillion"]}
    bone_idx = df.index[df["Domain group"] == "Bone mineral regional"]
    if len(bone_idx):
        hl[int(bone_idx[0])] = OI["sky"]
    _render_table(
        df, list(df.columns),
        "T4. Grouped permutation importance — all 16 domain groups",
        "T4_full_group_importance",
        col_widths=[0.7, 2.6, 1.0, 1.2, 2.2, 0.9, 1.3],
        highlight_rows=hl,
        fontsize=9.5, title_fs=13,
        note="Positive drop = group helps prediction. Vermillion = top contributor; sky = Bone mineral regional.",
        fig_w=11.5,
    )


# ---------------------------------------------------------------- T5 width sweep
def t5():
    s = pd.read_csv(os.path.join(HERE, "tfm_width_sweep_summary.csv"))
    rows = []
    for _, r in s.iterrows():
        w = int(r["width"])
        label = f"{w} (full)" if w == 628 else str(w)
        sd = f' ± {r["spearman_sd"]:.3f}' if not pd.isna(r["spearman_sd"]) else ""
        rows.append([
            label,
            int(r["n"]),
            f'{r["spearman_mean"]:+.3f}{sd}',
            f'{r["negmae_mean"]:.5f}',
        ])
    df = pd.DataFrame(rows, columns=["Feature width", "Draws",
                                     "LOOCV Spearman (mean ± SD)", "−MAE (mean)"])
    df.to_csv(os.path.join(HERE, "T5_width_sweep.csv"), index=False)
    hl = {len(rows) - 1: OI["blue"]}
    _render_table(
        df, list(df.columns),
        "T5. Feature-width sweep — signal emerges only at full width",
        "T5_width_sweep",
        col_widths=[1.6, 0.9, 3.0, 1.8],
        highlight_rows=hl,
        note=("Random feature subsets (5 draws each for width 5–250); 628 = the complete everything table "
              "(single draw). Signal is a whole-table property, not recoverable from a narrow subset."),
        fig_w=10.5, fontsize=10.5,
    )


if __name__ == "__main__":
    t1(); t2(); t3(); t4(); t5()
    print("done")
