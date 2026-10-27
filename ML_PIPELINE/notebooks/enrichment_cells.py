# %% [markdown]
# ## GO Enrichment
#
# Which biological processes are behind the genes each approach found?
#
# - **ORA (over-representation)** asks whether a gene list (ML consensus, DESeq2 significant, their overlap)
#   contains more genes from a GO term than expected by chance.
# - **GSEA** uses every gene, ranked: DESeq2 by direction and strength of change (signed −log10 p), and the ML
#   ensemble by mean importance rank. It finds terms whose genes cluster at the top (or, for DESeq2, the bottom)
#   of the ranking, without needing a cutoff.
#
# The background is the set of genes the pipeline analysed (the CV-filtered genes), not the whole genome, so
# terms aren't reported as enriched just because highly variable genes were selected.
# Settings come from `payload["config"]["enrichment"]`.

# %%
ENRICHMENT_DEFAULTS = {"enabled": False, "methods": ["ora", "gsea"], "ontologies": ["BP"],
                       "ora_lists": ["ml_consensus", "deseq2_significant", "overlap"],
                       "gsea_rankings": ["deseq2", "ml_importance"], "top_n_terms": 20}
enrich_cfg = {**ENRICHMENT_DEFAULTS, **payload["config"].get("enrichment", {})}
enrichment = None

def _build_enrichment_inputs():
    """Gene lists, rankings and universe from this run's results. Returns (lists, rankings, universe, notes)."""
    notes, lists, ranks = [], {}, {}
    de = deseq2_df.copy() if "deseq2_df" in globals() and not deseq2_df.empty else None
    padj_thr = globals().get("PADJ_THRESHOLD", 0.05)
    lfc_thr = globals().get("LOG2FC_THRESHOLD", 0.0)

    if de is not None:
        de["gene_id"] = de["gene_id"].astype(str)
        sig = de[(de["padj"] < padj_thr) & (de["log2_fold_change"].abs() > lfc_thr)]
        lists["deseq2_significant"] = sig["gene_id"].tolist()
        lists["deseq2_up"] = sig.loc[sig["log2_fold_change"] > 0, "gene_id"].tolist()
        lists["deseq2_down"] = sig.loc[sig["log2_fold_change"] < 0, "gene_id"].tolist()
        p = de["pvalue"].astype(float).clip(lower=1e-300)
        signed = np.sign(de["log2_fold_change"].astype(float)) * -np.log10(p)
        ok = np.isfinite(signed)
        ranks["deseq2"] = {"scores": dict(zip(de.loc[ok, "gene_id"], signed[ok].astype(float))), "score_type": "std"}
    else:
        notes.append("no DESeq2 results: skipping DESeq2 lists and ranking")

    if "consensus_genes" in globals():
        lists["ml_consensus"] = consensus_genes["gene"].astype(str).tolist()
    else:
        notes.append("no ML consensus genes: run the consensus cell first")
    if "rankings" in globals() and rankings:
        mean_rank = pd.concat(rankings.values(), axis=1).mean(axis=1).dropna()
        n = len(mean_rank)
        score = (n - mean_rank + 1) / n                     # best gene ~1, worst ~0; all positive
        ranks["ml_importance"] = {"scores": {str(g): float(s) for g, s in score.items()}, "score_type": "pos"}

    if "deseq2_significant" in lists and "ml_consensus" in lists:
        lists["overlap"] = (cmp["overlap"] if "cmp" in globals()
                            else sorted(set(lists["ml_consensus"]) & set(lists["deseq2_significant"])))

    if de is not None:
        universe = de["gene_id"].tolist()                   # the genes both analyses were given
    elif "ml_importance" in ranks:
        universe = list(ranks["ml_importance"]["scores"])
    else:
        universe = []
    return lists, ranks, universe, notes


if not result:
    print("Run the pipeline first")
elif not enrich_cfg["enabled"]:
    print('Enrichment is off. Set "enrichment": {"enabled": True, ...} in the config cell to run it.')
else:
    lists, ranks, universe, notes = _build_enrichment_inputs()
    cfg = dict(enrich_cfg)
    # Drop anything this run couldn't produce, rather than failing the whole request
    if "ora" in cfg["methods"]:
        unavailable = [n for n in cfg["ora_lists"] if n not in lists]
        cfg["ora_lists"] = [n for n in cfg["ora_lists"] if n in lists]
        notes += [f"ORA on '{n}' skipped: not available in this run" for n in unavailable]
    if "gsea" in cfg["methods"]:
        unavailable = [n for n in cfg["gsea_rankings"] if n not in ranks]
        cfg["gsea_rankings"] = [n for n in cfg["gsea_rankings"] if n in ranks]
        notes += [f"GSEA on '{n}' skipped: not available in this run" for n in unavailable]
    cfg["methods"] = [m for m in cfg["methods"]
                      if (m == "ora" and cfg["ora_lists"]) or (m == "gsea" and cfg["gsea_rankings"])]
    for n in notes:
        print("Note:", n)

    if not cfg["methods"] or not universe:
        print("Nothing to run for enrichment.")
    else:
        body = {"config": cfg, "universe": universe,
                "gene_lists": {n: lists[n] for n in cfg["ora_lists"]} if "ora" in cfg["methods"] else {},
                "rankings": {n: ranks[n] for n in cfg["gsea_rankings"]} if "gsea" in cfg["methods"] else {},
                "analysis_id": (result or {}).get("pipeline_id")}
        sizes = ", ".join(f"{n}: {len(g)}" for n, g in body["gene_lists"].items())
        print(f"Running GO {'/'.join(cfg['ontologies'])} enrichment ({', '.join(cfg['methods'])}) "
              f"on {len(universe)} background genes. Lists: {sizes or 'none'}. "
              f"Rankings: {', '.join(body['rankings']) or 'none'}.")
        print("This can take a few minutes (merging similar GO terms is the slow part)...")
        try:
            r = requests.post(f"{ORCHESTRATION_URL}/api/enrichment", json=body, timeout=1800)
            if r.status_code != 200:
                print(f"✗ Enrichment failed ({r.status_code}): {r.json().get('detail', r.text)[:800]}")
            else:
                enrichment = r.json()
                print(f"✓ Done: {len(enrichment['terms'])} enriched terms (organism: {enrichment['organism']})")
        except requests.exceptions.RequestException as e:
            print(f"✗ Couldn't reach the backend for enrichment: {e}")

if enrichment:
    summary_tbl = pd.DataFrame(enrichment["summaries"])
    display(summary_tbl.rename(columns={"n_input": "genes in", "n_mapped": "mapped to GO", "n_terms": "terms"}))

# %%
# Over-representation results: top terms per gene list
def _short_genes(genes, k=8):
    return ", ".join(genes[:k]) + (f" … (+{len(genes) - k})" if len(genes) > k else "")

if enrichment:
    terms = pd.DataFrame(enrichment["terms"])
    top_n = enrich_cfg.get("top_n_terms", 20)
    ora = terms[terms["method"] == "ora"] if not terms.empty else terms
    if ora.empty:
        print("No GO terms passed the cutoffs in the over-representation analysis.")
    for (name, ont), grp in (ora.groupby(["analysis", "ontology"], sort=False) if not ora.empty else []):
        grp = grp.sort_values("padj").head(top_n).copy()
        list_size = int(grp["gene_ratio"].iloc[0].split("/")[1])
        grp["ratio"] = grp["count"] / list_size
        print(f"\n{name} — GO:{ont}  ({len(grp)} of {int((ora['analysis'] == name).sum())} terms shown)")
        display(grp[["term_id", "description", "count", "gene_ratio", "bg_ratio", "padj", "genes"]]
                .assign(genes=grp["genes"].map(_short_genes))
                .style.format({"padj": "{:.2e}"}).hide(axis="index"))

        g = grp.iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(2.5, 0.35 * len(g) + 1)))
        sc = ax.scatter(g["ratio"], g["description"].str.slice(0, 60), s=g["count"] * 18,
                        c=-np.log10(g["padj"].clip(lower=1e-300)), cmap="viridis", edgecolor="none")
        cb = fig.colorbar(sc, ax=ax, pad=0.01); cb.set_label("−log10 adj. p")
        ax.set_xlabel(f"Gene ratio (genes in term / {list_size} {name} genes)")
        ax.set_title(f"GO:{ont} over-representation: {name}", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout(); plt.show()

# %%
# GSEA results: terms at the top (and bottom) of each ranking
if enrichment:
    gsea = terms[terms["method"] == "gsea"] if not terms.empty else terms
    if gsea.empty:
        print("No GO terms passed the cutoff in GSEA.")
    for (name, ont), grp in (gsea.groupby(["analysis", "ontology"], sort=False) if not gsea.empty else []):
        signed = (grp["nes"] < 0).any()
        if signed:   # show the strongest terms in each direction
            up = grp[grp["nes"] > 0].sort_values("padj").head(top_n // 2)
            down = grp[grp["nes"] < 0].sort_values("padj").head(top_n - len(up))
            grp = pd.concat([up, down])
        else:
            grp = grp.sort_values("padj").head(top_n)
        grp = grp.sort_values("nes")
        print(f"\n{name} — GO:{ont}  ({len(grp)} terms shown)")
        display(grp.sort_values("padj")[["term_id", "description", "set_size", "nes", "padj", "count", "genes"]]
                .assign(genes=grp["genes"].map(_short_genes))
                .rename(columns={"count": "leading edge"})
                .style.format({"nes": "{:.2f}", "padj": "{:.2e}"}).hide(axis="index"))

        fig, ax = plt.subplots(figsize=(9, max(2.5, 0.35 * len(grp) + 1)))
        colors = ["#c0392b" if v > 0 else "#2471a3" for v in grp["nes"]]
        ax.barh(grp["description"].str.slice(0, 60), grp["nes"], color=colors)
        ax.axvline(0, color="grey", lw=0.8)
        if name == "deseq2":
            fl = payload["config"].get("factor_values") or ["control", "treatment"]
            ax.set_xlabel(f"Normalized enrichment score  (red: higher in {fl[-1]}, blue: higher in {fl[0]})")
        else:
            ax.set_xlabel("Normalized enrichment score  (genes ranked by ML importance)")
        ax.set_title(f"GO:{ont} GSEA: {name}", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout(); plt.show()

# %%
# Save all enriched terms
if enrichment:
    out_dir = globals().get("RESULTS_DIR", Path("."))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = Path(out_dir) / f"go_enrichment_{stamp}.csv"
    pd.DataFrame(enrichment["terms"]).assign(genes=lambda d: d["genes"].map("/".join)).to_csv(fn, index=False)
    print(f"✓ Saved {len(enrichment['terms'])} terms to {fn}")
