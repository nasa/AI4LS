# ENVIRONMENTS.md — Exact Run Record

**Status:** verified against saved run logs and result files. Where the record is incomplete, the gap is stated rather than filled in.

## Software

| Component | Version | Evidence |
|---|---|---|
| Python | 3.11.13 | battery run log |
| tabpfn | **8.2.0** (documented install, 2026-08-16) | battery run log; pinned in `requirements.txt` |
| tabpfn (later reproduction) | 8.3.0 | reproduction session matched the headline correlation to 7 decimal places (expected 0.3762039, observed 0.3762038995750089) |
| torch | 2.13.0+cu130 | battery run log |

**Known gap, flagged not filled:** the battery result JSON files record the tabpfn version field as `null`. The 8.2.0 version comes from the run log's install record, not from the JSONs. The phase-5 nulls JSON records tabpfn **8.3.0** — phase 5 ran under the later 8.3.0 install — and the 8.3.0 reproduction of the headline matched to 7 decimal places, so the version split has no known effect on any number.

## Hardware

| Run | Machine |
|---|---|
| Registered battery (all phases) | NVIDIA A10G GPU |
| Earlier exploratory runs | NVIDIA T4 (Colab) and A10G, as noted per script header |

## Model weights

TabPFN-3 weights are gated and distributed under a non-commercial research license. They are **not** committed to this repository. Recorded checksums (SHA-256, first 16 hex digits) of the weights used:

| Checkpoint | SHA-256 prefix |
|---|---|
| `tabpfn-v3-regressor-v3_default.ckpt` | `311ce18d97e9533d` |
| `tabpfn-v3-classifier-v3_default.ckpt` | `d0d865d54dfbc524` |

To run anything here, register with PriorLabs and `export TABPFN_TOKEN="your_token_here"`.

## Seeds and determinism

| Scope | Policy |
|---|---|
| Model | seed 42, `n_estimators=32`, deterministic algorithms on |
| LOOCV fold order | deterministic, subject index 0..37 |
| Battery permutation nulls | `seed[i] = i`, recorded per row of each null CSV |
| Phase-5 null B1 (fold-safe regression) | `seed[i] = 42*1000 + i` |
| Phase-5 null B2 (A-core CONTROL vs FLY labels) | `seed[i] = 42*2000 + i` |

## Input fingerprint

Every battery result JSON records the MD5 of the modeling table it ran on: `ffd3c4fec75ab2d627985e03c162dddb` (`data/c11_totalhipBMD_change_features_all.csv`). If the file changes, the fingerprint changes, and the results no longer claim to describe it.
