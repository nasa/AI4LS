import h5py
import numpy as np

def decode(val):
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8").strip()
    return str(val).strip()

TISSUE_MAP = {
    "Left Lobe of the Liver":          "Liver",
    "Left lobe of liver":              "Liver",
    "liver":                           "Liver",
    "Left kidney":                     "Kidney",
    "Right kidney":                    "Kidney",
    "Left Lung":                       "Lung",
    "Right Lung":                      "Lung",
    "Left ventricle":                  "Heart",
    "Right ventricle":                 "Heart",
    "heart right ventricle":           "Heart",
    "Right retina":                    "Retina",
    "Left retina":                     "Retina",
    "thymus":                          "Thymus",
    "Right soleus":                    "Soleus",
    "Soleus-both sides":               "Soleus",
    "Right gastrocnemius":             "Gastrocnemius",
    "Left gastrocnemius":              "Gastrocnemius",
    "Left quadriceps femoris":         "Quadriceps",
    "Right quadriceps femoris":        "Quadriceps",
    "Quadriceps femoris":              "Quadriceps",
    "Right tibialis anterior":         "Tibialis",
    "Left tibialis anterior":          "Tibialis",
    "Right extensor digitorum longus": "EDL",
    "Extensor digitorum longus- both sides": "EDL",
    "right hemisphere of cerebellum":  "Cerebellum",
    "Left cerebral hemisphere":        "Brain",
    "Cerebrum":                        "Brain",
    "brain":                           "Brain",
    "Right hippocampus":               "Hippocampus",
    "descending colon":                "Colon",
    "dorsal skin":                     "Skin",
    "femoral skin":                    "Skin",
    "femoral lateral skin":            "Skin",
    "white adipose tissue":            "Adipose",
    "brown adipose tissue":            "Adipose",
    "adrenal gland":                   "Adrenal Gland",
    "Adrenal glands- both sides":      "Adrenal Gland",
    "Adrenal gland":   "Adrenal Gland",
    "mammary gland":   "Mammary Gland",
    "Spleen-distal":                   "Spleen",
    "Whole Spleen":                    "Spleen",
    "left eye":                        "Eye",
    "eye":                             "Eye",
    "Right optic nerve":               "Optic Nerve",
    "Cortical Bone":                   "Bone",
    "Temporal Bone":                   "Bone",
    "Mandible":                        "Bone",
    "Cells":                           "Unknown",
    "Cells, Cultured":                 "Unknown",
    "Zygote":                          "Unknown",
    "Trp53 null Mammary Tumor":        "Unknown",
    "Tissue":                          "Unknown",
    "":                                "Unknown",
}

with h5py.File("OSDR_mouse_RNAseq_Feb2026.h5", "r") as f:
    tissue_raw = f["meta"]["samples"]["characteristics"]["study.characteristics.material type"][:]


decoded  = np.array([decode(v) for v in tissue_raw])
harmonized = np.array([TISSUE_MAP.get(v, v) for v in decoded])


MIN_COUNT = 30
unique, counts = np.unique(harmonized, return_counts=True)
count_dict = dict(zip(unique, counts))

# collapse low-count and Unknown to Other
harmonized = np.array([
    t if (count_dict.get(t, 0) >= MIN_COUNT and t != "Unknown") else "Other"
    for t in harmonized
])

# verify final counts
unique, counts = np.unique(harmonized, return_counts=True)
print(f"\nFinal tissue categories: {len(unique)}")
for c, u in sorted(zip(counts, unique), reverse=True):
    print(f"  {int(c):4d}  {u}")

