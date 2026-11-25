"""
Runs one CRISP analysis in its own process (started by service.py):

    python -m src.crisp_runner <workdir>/job.json

Reads the job, prepares the data (prepare.py), runs CRISP's main.run() unchanged except for
the LIRM patch noted in crisp/main.py, normalizes each method's output, ranks the genes
(ranking.py) and writes <workdir>/output.json. A separate process keeps CRISP's memory, torch
threads and any hang out of the gRPC server, and lets the server enforce a timeout.
"""
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent
CRISP_DIR = APP_DIR / "crisp"
sys.path.insert(0, str(CRISP_DIR))
sys.path.insert(0, str(APP_DIR))

# CRISP imports its Google Cloud helpers at module level; the pipeline never uses the cloud path,
# so give it a stand-in instead of installing google-cloud-storage.
_gcp = types.ModuleType("utils.gcp_helpers")
def _no_cloud(*a, **k):
    raise RuntimeError("cloud storage is not available in causal_service")
for _name in ("save_json_to_bucket", "save_dataframe_to_bucket", "get_json_from_bucket",
              "get_dataframe_from_bucket"):
    setattr(_gcp, _name, _no_cloud)
sys.modules["utils.gcp_helpers"] = _gcp
os.environ.setdefault("MPLBACKEND", "Agg")

from src.prepare import prepare, effect_sizes, NORMALIZATION          # noqa: E402
from src.ranking import consensus_ranking, crisp_v11_ranking, CAUSAL_CODES   # noqa: E402

DEFAULT_MODELS = ["ICP", "NLICP", "IRM", "LIRM"]
KNOWN_MODELS = {"ICP", "NLICP", "IRM", "LIRM", "ERM", "RF"}
METHOD_CODES = {
    "Linear ICP": "ICP", "Non-Linear ICP": "NLICP", "Linear IRM": "LIRM", "Non-Linear IRM": "IRM",
    "Non Linear IRM (Feature Reduction Mode)": "FRIRM", "Non-Causal ERM": "ERM", "RF": "RF",
}


def _flat(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return [float(x)]
    x = list(x)
    if len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = list(x[0])
    return [float(v) if v is not None else float("nan") for v in x]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def normalize_method(r: dict) -> dict:
    """One results_for_bucket entry -> a uniform record."""
    name = r["method"]
    code = METHOD_CODES.get(name, name)
    features = [str(f) for f in r.get("features") or []]
    coefs = _flat(r.get("coefficients"))
    note = ""
    if coefs is None:
        coefs, note = [], "no solution (no invariant set accepted)" if code in ("ICP", "NLICP") else "no coefficients"
    elif len(coefs) != len(features):
        note = f"{len(coefs)} coefficients for {len(features)} features; coefficients dropped"
        coefs = []
    p = r.get("pvals")
    pvals, pvalue = [], float("nan")
    if isinstance(p, (list, tuple)):
        flat = _flat(p)
        if flat and len(flat) == len(features):
            pvals = flat
        elif flat and len(flat) == 1:
            pvalue = flat[0]
    elif p is not None:
        pvalue = _num(p)
    selection = r.get("selection") or ""
    if code in ("ICP", "NLICP"):
        # ICP's p-values test whether a gene SET is invariant (high = accepted), so they are not
        # per-gene significance and are not reported as pvals
        pvals = []
        if selection == "defining_set":
            note = (note + "; " if note else "") + ("no gene is in every accepted subset; these genes come from "
                                                    "CRISP's best-fit fallback and are not significant")
        elif selection == "none":
            note = note or "no gene subset passed the invariance test"
    return {"method": name, "code": code, "causal": code in CAUSAL_CODES, "features": features,
            "coefficients": coefs, "pvals": pvals, "pvalue": pvalue,
            "test_acc": _num(r.get("test_acc")), "val_acc": _num(r.get("validate_acc")), "note": note,
            "selection": selection,
            "n_tested_subsets": int(r.get("n_tested_subsets") or 0),
            "n_accepted_subsets": int(r.get("n_accepted_subsets") or 0)}


def significant_genes(methods):
    """gene -> methods whose formal ICP result (the intersection of all accepted subsets) contains it.
    Under ICP's assumptions that set holds only invariant genes with probability >= 1 - alpha."""
    sig = {}
    for m in methods:
        if m["code"] in ("ICP", "NLICP") and m["selection"] == "intersection":
            for g in m["features"]:
                sig.setdefault(g, []).append(m["code"])
    return sig


def build_config(job: dict, data_fp: Path, results_dir: Path) -> dict:
    seed = int(job.get("seed") or 12)
    models = [m.upper() for m in (job.get("models") or DEFAULT_MODELS)]
    unknown = set(models) - KNOWN_MODELS
    if unknown:
        raise ValueError(f"Unknown CRISP models {sorted(unknown)}; supported: {sorted(KNOWN_MODELS)}")
    options = json.loads(job.get("model_options_json") or "{}")
    ensemble = {"models": models}
    for code in ("ICP", "NLICP", "IRM", "LIRM", "ERM", "FRIRM"):
        ensemble[code] = {"seed": seed, **options.get(code, {})}
    if "RF" in options:
        ensemble["RF"] = options["RF"]
    return {
        "name": f"crisp_{job['analysis_id']}", "short_name": f"crisp_{job['analysis_id']}",
        "verbose": 1, "use_cloud": False,
        "test_val_split": [float(job.get("test_fraction") or 0.2), float(job.get("val_fraction") or 0.1)],
        "data_options": {
            "dataset_fp": str(data_fp), "subject_keys": "sample", "targets": ["target"],
            "predictors": "All", "environments": ["env"], "exclude": ["sample", "env"],
            "output_data_regime": "binary",
        },
        "feature_selection_options": {"max_features": int(job.get("max_features") or 20), "verbose": 0,
                                      "seed": seed},
        "ensemble_options": ensemble,
        "results_directory": str(results_dir) + "/",
    }


def main(job_path: str):
    t0 = time.time()
    job_path = Path(job_path)
    workdir = job_path.parent
    job = json.loads(job_path.read_text())
    out_path = workdir / "output.json"
    try:
        seed = int(job.get("seed") or 12)
        random.seed(seed)
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)

        df = pd.read_csv(workdir / "input.csv.gz", index_col=0)
        crisp_df, environments, notes, log_cpm = prepare(df, job["group_column"], job["control_value"],
                                                job["treatment_value"], job["environment_column"])
        data_fp = workdir / "crisp_input.pkl"
        crisp_df.to_pickle(data_fp)
        results_dir = workdir / "results"
        results_dir.mkdir(exist_ok=True)
        config = build_config(job, data_fp, results_dir)
        (workdir / "crisp_config.json").write_text(json.dumps(config, indent=2))
        n_genes = crisp_df.shape[1] - 3
        print(f"CRISP input: {len(crisp_df)} samples x {n_genes} genes, environments "
              f"{[e['name'] for e in environments if e['used']]}, models {config['ensemble_options']['models']}",
              flush=True)

        import main as crisp_main
        crisp_main.run(config)

        raw = json.loads((results_dir / "results_for_bucket.json").read_text())["results"]
        methods = [normalize_method(r) for r in raw if "method" in r]
        ranking_method = job.get("ranking_method") or "consensus"
        if ranking_method == "consensus":
            top_k = int(job.get("top_k") or config["feature_selection_options"]["max_features"])
            ranking = consensus_ranking(methods, top_k)
        elif ranking_method == "crisp_v1.1":
            ranking = crisp_v11_ranking(raw)
        else:
            raise ValueError(f"Unknown ranking_method '{ranking_method}' (use consensus or crisp_v1.1)")
        # significance: genes in ICP's or NLICP's formal intersection; add any the ranking missed
        sig = significant_genes(methods)
        missing = [g for g in sig if g not in set(ranking["gene"])]
        if missing:
            ranking = pd.concat([ranking, pd.DataFrame({
                "gene": missing, "rank": 0, "score": 0.0, "votes": 0,
                "n_methods": int(ranking["n_methods"].iloc[0]) if len(ranking) else 0,
                "consensus": False})], ignore_index=True)
            ranking["rank"] = np.arange(1, len(ranking) + 1)
        ranking["significant"] = ranking["gene"].isin(set(sig))
        ranking["significant_in"] = ranking["gene"].map(lambda g: ",".join(sig.get(g, [])))
        for m in methods:
            if m["code"] in ("ICP", "NLICP"):
                notes.append(f"{m['method']}: {m['selection'] or 'n/a'} "
                             f"({m['n_accepted_subsets']} of {m['n_tested_subsets']} gene subsets accepted as invariant)")
        # direction from the data: +1 higher with the treatment value, -1 lower
        eff = effect_sizes(log_cpm, ranking["gene"].tolist())
        ranking = ranking.merge(eff, on="gene", how="left")
        ranking["direction"] = np.sign(ranking["log2fc"].fillna(0)).astype(int)
        ranking.to_csv(workdir / "ranking.csv", index=False)
        for m in methods:
            if m["causal"] and not m["coefficients"]:
                notes.append(f"{m['method']}: {m['note']}")
        out = {"success": True, "ranking_method": ranking_method, "methods": methods,
               "n_significant": int(ranking["significant"].sum()),
               "ranking": ranking.to_dict(orient="records"), "environments": environments,
               "n_samples": int(len(crisp_df)), "n_genes": int(n_genes), "normalization": NORMALIZATION,
               "notes": notes, "runtime_seconds": time.time() - t0}
    except Exception as e:
        import traceback
        traceback.print_exc()
        out = {"success": False, "error_message": f"{type(e).__name__}: {e}", "runtime_seconds": time.time() - t0}
    out_path.write_text(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0 if out["success"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
