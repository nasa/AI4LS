"""
Turn a pipeline dataset (raw counts + metadata, one row per sample) into CRISP's input table:
    sample | env | target | gene_1 ... gene_n
  - target: 1 = treatment value (e.g. Space Flight), 0 = control value; other samples are dropped
  - env:    the environment column (OSD study); studies without both groups are dropped
  - genes:  log2(CPM + 1), then z-scored per gene (CRISP expects normalized continuous predictors)
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

METADATA_PREFIXES = ("Factor Value", "Characteristics", "Comment", "Parameter Value")
NON_GENE_COLUMNS = {"source_dataset", "sample", "env", "target"}
NORMALIZATION = "log2(CPM + 1), z-scored per gene"


def gene_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    return [c for c in df.select_dtypes(include="number").columns
            if c not in exclude and c not in NON_GENE_COLUMNS and not str(c).startswith(METADATA_PREFIXES)]


def prepare(df: pd.DataFrame, group_column: str, control: str, treatment: str,
            environment_column: str, min_per_group: int = 1) -> Tuple[pd.DataFrame, List[Dict], List[str], pd.DataFrame]:
    """Returns (crisp_df, environments, notes, log_cpm) where log_cpm is the log2(CPM + 1) table
    (before z-scoring) with sample/env/target columns, for effect sizes.
    Raises ValueError when CRISP can't run on this data."""
    notes = []
    for col in (group_column, environment_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in dataset (columns include: {', '.join(map(str, df.columns[:8]))}, ...)")

    groups = df[group_column].astype(str)
    keep = groups.isin([control, treatment])
    if (~keep).any():
        notes.append(f"{int((~keep).sum())} samples with other {group_column} values dropped")
    df = df.loc[keep]
    target = (df[group_column].astype(str) == treatment).astype(int)
    env = df[environment_column].astype(str)

    environments, used = [], []
    for name in sorted(env.unique()):
        t = target[env == name]
        n_t, n_c = int(t.sum()), int((t == 0).sum())
        ok = n_t >= min_per_group and n_c >= min_per_group
        environments.append({"name": name, "n_control": n_c, "n_treatment": n_t, "used": ok,
                             "note": "" if ok else f"needs both {control} and {treatment} samples"})
        if ok:
            used.append(name)
    if len(used) < 2:
        raise ValueError(f"CRISP needs at least 2 environments ({environment_column} values) with both "
                         f"{control} and {treatment} samples; usable: {used or 'none'}. "
                         f"Environments: {[(e['name'], e['n_control'], e['n_treatment']) for e in environments]}")
    dropped = [e["name"] for e in environments if not e["used"]]
    if dropped:
        notes.append(f"Environments dropped (missing a group): {', '.join(dropped)}")
    mask = env.isin(used)
    df, target, env = df.loc[mask], target[mask], env[mask]

    genes = gene_columns(df, exclude=[group_column, environment_column])
    if not genes:
        raise ValueError("No numeric gene columns found")
    X = df[genes].astype(float).clip(lower=0)
    lib = X.sum(axis=1).replace(0, np.nan)
    X = np.log2(X.div(lib, axis=0) * 1e6 + 1)
    log_cpm = X.copy()
    sd = X.std(axis=0, ddof=0)
    X = (X - X.mean(axis=0)) / sd.replace(0, np.nan)
    X = X.fillna(0.0)          # zero-variance genes -> all 0; CRISP's zero-variance check removes them

    samples = pd.Series([str(s) for s in df.index], index=df.index)
    if samples.duplicated().any():
        samples = samples + "_" + samples.groupby(samples).cumcount().astype(str)
    out = pd.concat([pd.DataFrame({"sample": samples.values, "env": env.values, "target": target.values},
                                  index=df.index), X.astype(float)], axis=1).reset_index(drop=True)
    out.columns = [str(c) for c in out.columns]
    if out.columns.duplicated().any():
        raise ValueError(f"Duplicate gene names: {sorted(set(out.columns[out.columns.duplicated()]))[:10]}")
    log_cpm.columns = [str(c) for c in log_cpm.columns]
    log_cpm = log_cpm.reset_index(drop=True).assign(env=out["env"].values, target=out["target"].values)
    return out, environments, notes, log_cpm


def effect_sizes(log_cpm: pd.DataFrame, genes: List[str]) -> pd.DataFrame:
    """Per gene: log2fc = mean log2(CPM+1) treatment - control (all samples), and whether the
    difference has the same sign in every environment."""
    rows = []
    for g in genes:
        if g not in log_cpm.columns:
            rows.append({"gene": g, "log2fc": float("nan"), "consistent": False})
            continue
        v, t = log_cpm[g], log_cpm["target"]
        fc = float(v[t == 1].mean() - v[t == 0].mean())
        per_env = log_cpm.groupby("env").apply(
            lambda e: e.loc[e["target"] == 1, g].mean() - e.loc[e["target"] == 0, g].mean())
        signs = np.sign(per_env.values)
        rows.append({"gene": g, "log2fc": fc,
                     "consistent": bool(len(signs) > 0 and np.all(signs == signs[0]) and signs[0] != 0)})
    return pd.DataFrame(rows, columns=["gene", "log2fc", "consistent"])
