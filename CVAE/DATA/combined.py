import h5py
import numpy as np

with h5py.File("subset_final.h5", "r") as f:
    symbols = np.array([v.decode() for v in f["meta/genes/symbol"][:]])
    expr    = f["data/expression"][:].T   # (samples, genes)

# symbol filter
def is_characterized(s):
    if s.startswith("Gm") and s[2:].split("-")[0].isdigit():
        return False
    if "-ps" in s:
        return False
    if "Rik" in s:
        return False
    return True

symbol_mask     = np.array([is_characterized(s) for s in symbols])
expression_mask = (expr > 1).mean(axis=0) >= 0.10
combined_mask   = symbol_mask & expression_mask

print(f"Symbol filter only:     {symbol_mask.sum():5d} genes kept")
print(f"Expression filter only: {expression_mask.sum():5d} genes kept")
print(f"Combined:               {combined_mask.sum():5d} genes kept")
print(f"Total removed:          {(~combined_mask).sum():5d} genes")
