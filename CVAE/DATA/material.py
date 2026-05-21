import h5py
import numpy as np

def decode(val):
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8").strip()
    return str(val).strip()

with h5py.File("OSDR_mouse_RNAseq_Feb2026.h5", "r") as f:
    #tissue_raw = f["meta"]["samples"]["characteristics"]["study.characteristics.material type"][:]
    #tissue_raw = f["meta"]["samples"]["characteristics"]["study.characteristics.age"][:]
    tissue_raw = f["meta"]["samples"]["parameters"]['study.parameter value.euthanasia method'][:]
    decoded = np.array([decode(v) for v in tissue_raw])
    
    n_empty = (decoded == "").sum()
    unique, counts = np.unique(decoded, return_counts=True)
    
    print(f"Missing: {n_empty} / {len(decoded)}")
    print(f"\nUnique values ({len(unique)} total):")
    for u, c in sorted(zip(counts, unique), reverse=True):
        print(f"  {u}: {c}")
