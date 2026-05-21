import h5py
with h5py.File("subset_final.h5", "r") as f:
    expr = f["data/expression"][:].T   # (samples, genes)

# fraction of samples with count > 1
expressed_frac = (expr > 1).mean(axis=0)

# standard cutoff: expressed in at least 10% of samples
keep_mask = expressed_frac >= 0.10
print(f"Genes passing expression filter: {keep_mask.sum()} / {len(keep_mask)}")
print(f"Removed: {(~keep_mask).sum()}")
