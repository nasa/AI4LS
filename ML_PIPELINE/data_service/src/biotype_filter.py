"""
biotype_filter.py  ->  data_service/src/biotype_filter.py

Keep only genes of chosen biotypes (default: protein_coding) using a local GTF.

Works with Ensembl GTFs (gene_biotype) and GENCODE GTFs (gene_type), plain or .gz.
Matches dataset columns by Ensembl gene ID (version suffixes like .4 ignored) and,
for columns that are already gene symbols, by gene_name.

Reference location: every *.gtf / *.gtf.gz in GTF_DIR (default /app/reference) is loaded,
so a mouse and a human GTF can sit side by side; Ensembl IDs never collide across species.
The first parse of each GTF is cached as a small TSV next to it (or in /tmp if read-only),
so later runs start in about a second instead of re-reading a 50 MB GTF.
"""
from __future__ import annotations

import gzip
import logging
import os
import re
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

GTF_DIR = Path(os.environ.get("GTF_DIR", "/app/reference"))
DEFAULT_KEEP_BIOTYPES = ("protein_coding",)

_ENSEMBL_ID = re.compile(r"^ENS[A-Z]*G\d{11}(\.\d+)?$")
_ATTR = re.compile(r'(\w+) "([^"]*)"')

_lock = threading.Lock()
_annotation: Optional[pd.DataFrame] = None   # columns: gene_id, gene_name, biotype


def _strip_version(gene_id: str) -> str:
    return gene_id.split(".", 1)[0] if gene_id.startswith("ENS") else gene_id


def _parse_gtf(path: Path) -> pd.DataFrame:
    """Read gene records from one GTF: gene_id (unversioned), gene_name, biotype."""
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t", 8)
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = dict(_ATTR.findall(parts[8]))
            gid = attrs.get("gene_id")
            if not gid:
                continue
            rows.append((_strip_version(gid),
                         attrs.get("gene_name", ""),
                         attrs.get("gene_biotype") or attrs.get("gene_type") or "unknown"))
    if not rows:
        raise ValueError(f"No 'gene' records found in {path}. Is it a GTF with gene lines?")
    return pd.DataFrame(rows, columns=["gene_id", "gene_name", "biotype"]).drop_duplicates("gene_id")


def _cache_path(gtf: Path) -> Path:
    name = gtf.name + ".genes.tsv"
    beside = gtf.with_name(name)
    if os.access(gtf.parent, os.W_OK):
        return beside
    return Path("/tmp") / name


def _load_one(gtf: Path) -> pd.DataFrame:
    cache = _cache_path(gtf)
    if cache.exists() and cache.stat().st_mtime >= gtf.stat().st_mtime:
        return pd.read_csv(cache, sep="\t", dtype=str, keep_default_na=False)
    logger.info(f"Parsing GTF {gtf} (first time only)...")
    df = _parse_gtf(gtf)
    try:
        df.to_csv(cache, sep="\t", index=False)
    except OSError as e:
        logger.warning(f"Couldn't write GTF cache {cache}: {e}")
    return df


def load_annotation(gtf_paths: Optional[Sequence[str | Path]] = None) -> pd.DataFrame:
    """All genes from the GTF(s). Loaded once per process unless explicit paths are given."""
    global _annotation
    if gtf_paths is None:
        with _lock:
            if _annotation is not None:
                return _annotation
            gtfs = sorted(list(GTF_DIR.glob("*.gtf")) + list(GTF_DIR.glob("*.gtf.gz")))
            if not gtfs:
                raise FileNotFoundError(
                    f"No .gtf or .gtf.gz files in {GTF_DIR}. Mount a folder with the GTF(s) there "
                    f"(or set GTF_DIR) to use the biotype filter.")
            _annotation = pd.concat([_load_one(g) for g in gtfs]).drop_duplicates("gene_id").reset_index(drop=True)
            logger.info(f"Loaded {len(_annotation):,} genes from {', '.join(g.name for g in gtfs)}")
            return _annotation
    return pd.concat([_load_one(Path(p)) for p in gtf_paths]).drop_duplicates("gene_id").reset_index(drop=True)


def classify_columns(columns: Iterable[str], annotation: pd.DataFrame) -> pd.DataFrame:
    """Biotype for each column: matched by Ensembl ID first, then by gene symbol."""
    by_id = dict(zip(annotation["gene_id"], annotation["biotype"]))
    named = annotation[annotation["gene_name"] != ""]
    # A symbol shared by several genes counts as coding if any of them is (e.g. a gene and its pseudogene copy)
    by_name = named.groupby("gene_name")["biotype"].agg(
        lambda b: "protein_coding" if "protein_coding" in set(b) else b.iloc[0]).to_dict()
    out = []
    for col in columns:
        c = str(col)
        if _ENSEMBL_ID.match(c):
            out.append((col, by_id.get(_strip_version(c)), "id"))
        else:
            base = re.sub(r"_\d+$", "", c)   # symbols de-duplicated as Gene_1, Gene_2 by the converter
            bt = by_name.get(c, by_name.get(base))
            out.append((col, bt, "name" if bt else None))
    return pd.DataFrame(out, columns=["column", "biotype", "matched_by"])


def filter_by_biotype(
    df: pd.DataFrame,
    keep_biotypes: Sequence[str] = DEFAULT_KEEP_BIOTYPES,
    protected_columns: Sequence[str] = (),
    keep_unannotated: bool = False,
    annotation: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Drop gene columns whose biotype is not in keep_biotypes.

    Non-numeric columns and protected_columns (e.g. the target/condition column) are always kept.
    Genes not found in the GTF are dropped unless keep_unannotated=True.
    Returns (filtered_df, report).
    """
    annotation = load_annotation() if annotation is None else annotation
    keep = set(keep_biotypes)
    protected = set(protected_columns)
    gene_cols = [c for c in df.columns
                 if c not in protected and pd.api.types.is_numeric_dtype(df[c])]
    other_cols = [c for c in df.columns if c not in set(gene_cols)]

    cls = classify_columns(gene_cols, annotation)
    unannotated = cls["biotype"].isna()
    kept_mask = cls["biotype"].isin(keep) | (unannotated & keep_unannotated)
    kept_genes = cls.loc[kept_mask, "column"].tolist()

    removed = cls.loc[~kept_mask]
    report = {
        "genes_before": len(gene_cols),
        "genes_after": len(kept_genes),
        "keep_biotypes": sorted(keep),
        "unannotated": int(unannotated.sum()),
        "unannotated_kept": bool(keep_unannotated),
        "removed_by_biotype": removed["biotype"].fillna("not in GTF").value_counts().to_dict(),
        "matched_by_symbol": int((cls["matched_by"] == "name").sum()),
    }
    if report["genes_before"] and report["unannotated"] / report["genes_before"] > 0.5:
        logger.warning(f"{report['unannotated']} of {report['genes_before']} genes aren't in the GTF. "
                       "Is it the right organism and annotation release for this dataset?")
    logger.info(f"Biotype filter: {report['genes_before']} → {report['genes_after']} genes "
                f"(removed {report['removed_by_biotype']})")
    # keep the original column order
    keep_set = set(kept_genes) | set(other_cols)
    return df[[c for c in df.columns if c in keep_set]], report
