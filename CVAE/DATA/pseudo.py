import h5py
import numpy as np

with h5py.File("subset_final.h5", "r") as f:
    symbols = np.array([v.decode() for v in f["meta/genes/symbol"][:]])

# check how many look like pseudogenes/uncharacterized
gm     = [s for s in symbols if s.startswith("Gm") and s[2:].split("-")[0].isdigit()]
ps     = [s for s in symbols if s.endswith("-ps") or "-ps" in s]
rik    = [s for s in symbols if "Rik" in s]

print(f"Gm* uncharacterized: {len(gm)}")
print(f"-ps pseudogenes:    {len(ps)}")
print(f"Rik clones:         {len(rik)}")
print(f"Total to remove:     {len(set(gm + ps + rik))}")
print(f"Current genes:       {len(symbols)}")
