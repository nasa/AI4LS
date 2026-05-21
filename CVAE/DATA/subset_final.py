import h5py
import numpy as np

def decode(val):
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8").strip()
    return str(val).strip()

def is_characterized(s):
    if s.startswith("Gm") and s[2:].split("-")[0].isdigit():
        return False
    if "-ps" in s:
        return False
    if "Rik" in s:
        return False
    return True

# --- all your existing maps ---
EUTHANASIA_MAP = {
    # Ketamine/Xylazine variants
    "Ketamine/Xylazine/Acepromazine, Cardiac Puncture":                         "Ketamine_Xylazine",
    "Ketamine/Xylazine/Acepromazine, Cardiac Puncture, Thoracotomy":            "Ketamine_Xylazine",
    "Ketamine/Xylazine/Acepromazine, Thoracotomy, Cardiac Puncture":            "Ketamine_Xylazine",
    "ketamine/xylazine/acepromazine":                                            "Ketamine_Xylazine",
    "Ketamine/xylazine IP followed by cervical dislocation":                     "Ketamine_Xylazine",
    "Ketamine/xylazine":                                                         "Ketamine_Xylazine",
    "ketamine/xylazine IP":                                                      "Ketamine_Xylazine",
    "intraperitoneal (IP) injection of Ketamine/Xylazine (150/45 mg/kg) anesthesia": "Ketamine_Xylazine",
    "Ketamine/Xylazine/Acepromazine, Cardiac Puncture, Thoracotomy":            "Ketamine_Xylazine",

    # Isoflurane variants
    "Isoflurane":                                                                "Isoflurane",
    "Isoflurane, Thoracotomy, Cardiac Puncture":                                 "Isoflurane",
    "Isoflurane, Cardiac Puncture, Thoracotomy":                                 "Isoflurane",
    "Isoflurane/Exsanguination":                                                 "Isoflurane",
    "Isoflurane-anesthetised mice were euthanized by exsanguination":            "Isoflurane",
    "Isoflurane-anesthetised and euthanized by exsanguination":                  "Isoflurane",
    "isoflurane anesthesia and euthanized by exsanguination/double thoracotomy": "Isoflurane",
    "All mice were euthanized by inhaling lethal doses of isoflurane":           "Isoflurane",
    "Bilateral thoracotomy with sedation, Inhalation of Isoflurane":             "Isoflurane",
    "Bilateral thoracotomy with sedation, Cardiac puncture, Inhalation of Isoflurane": "Isoflurane",
    "Bilateral thoracotomy with sedation, Cardiac puncture, Inhalation of Isoflurane cancel": "Isoflurane",

    # CO2 variants
    "Carbon Dioxide":                                                            "CO2",
    "Cervical dislocation with sedation, Inhalation of CO2":                    "CO2",
    "CO2 inhalation":                                                            "CO2",

    # Exsanguination
    "Exsanguination":                                                            "Exsanguination",
    "ketamine/xylazine overdose and exsanguination/double thoracotomy":          "Exsanguination",
    "gas anesthesia and euthanized by exsanguination/double thoracotomy":        "Exsanguination",

    # Cardiac/thoracotomy combinations (no primary agent specified)
    "Bilateral thoracotomy with sedation, Cardiac puncture, Ketamine/xylazine injection": "Ketamine_Xylazine",
    "Bilateral thoracotomy with sedation, Cardiac puncture with sedation, Ketamine/xylazine injection": "Ketamine_Xylazine",
    "Bilateral thoracotomy with sedation, Cardiac puncture with sedation, Gas anesthesia": "Isoflurane",

    # Other
    "Euthasol":                                                                  "Pentobarbital",
    "Pentobarbital":                                                             "Pentobarbital",
    "":                                                                          "Unknown",
}

FLIGHT_MAP = {
    "Space Flight":         1,
    "Ground Control":       0,
    "Ground Control Rerun": 0,
    "Ground control":       0,
    "Vivarium Control":     0,
    "Vivarium control":     0,
    "Basal Control":        0,
    "Cohort Control #1":    0,
    "Cohort Control #2":    0,
    "":                     None,
}
SEX_MAP = {
    "Female": "Female", "female": "Female",
    "Male":   "Male",   "male":   "Male",
    "":       "Unknown",
}
STRAIN_MAP = {
    "C57BL/6NTac":  "C57BL/6", "C57BL/6NCrl":  "C57BL/6",
    "BALB/cAnNTac": "BALB/c",
    "B6;129S2":     "B6129SF2", "B6129SF2/J":   "B6129SF2",
    "DBA/2 Mouse":  "DBA/2",
}
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
    "Adrenal gland":                   "Adrenal Gland",
    "Adrenal glands- both sides":      "Adrenal Gland",
    "Spleen-distal":                   "Spleen",
    "Whole Spleen":                    "Spleen",
    "left eye":                        "Eye",
    "eye":                             "Eye",
    "Right optic nerve":               "Optic Nerve",
    "Cortical Bone":                   "Bone",
    "Temporal Bone":                   "Bone",
    "Mandible":                        "Bone",
    "mammary gland":                   "Mammary Gland",
    "Cells":                           "Unknown",
    "Cells, Cultured":                 "Unknown",
    "Zygote":                          "Unknown",
    "Trp53 null Mammary Tumor":        "Unknown",
    "Tissue":                          "Unknown",
    "":                                "Unknown",
}

MIN_TISSUE_COUNT = 30

with h5py.File("OSDR_mouse_RNAseq_Feb2026.h5", "r") as fin:

    # --- Gene filtering ---
    biotype  = np.array([decode(v) for v in fin["meta"]["genes"]["biotype"][:]])
    ensembl  = np.array([decode(v) for v in fin["meta"]["genes"]["ensembl_gene"][:]])
    symbol   = np.array([decode(v) for v in fin["meta"]["genes"]["symbol"][:]])
    expr_all = fin["data"]["expression"][:]   # (53511, 3315)

    # step 1: protein-coding only
    pc_mask = biotype == "protein_coding"

    # step 2: symbol filter on protein-coding subset
    symbol_pc       = symbol[pc_mask]
    symbol_mask     = np.array([is_characterized(s) for s in symbol_pc])

    # step 3: expression filter — compute on protein-coding genes x filtered samples
    #         (need sample mask first, then apply expression filter)
    flight_raw  = fin["meta"]["samples"]["factors"]["study.factor value.spaceflight"][:]
    strain_raw  = fin["meta"]["samples"]["characteristics"]["study.characteristics.strain"][:]
    sex_raw     = fin["meta"]["samples"]["characteristics"]["study.characteristics.sex"][:]
    study_raw   = fin["meta"]["samples"]["characteristics"]["id.accession"][:]
    tissue_raw  = fin["meta"]["samples"]["characteristics"]["study.characteristics.material type"][:]
    euth_raw = fin["meta"]["samples"]["parameters"]["study.parameter value.euthanasia method"][:]


    flight_labels  = np.array([FLIGHT_MAP.get(decode(v), None) for v in flight_raw])
    strain_decoded = np.array([decode(v) for v in strain_raw])
    sample_mask    = np.array([
        flight_labels[i] is not None and strain_decoded[i] != ""
        for i in range(len(flight_labels))
    ])

    # expression filter on protein-coding genes x valid samples
    expr_pc      = expr_all[np.ix_(pc_mask, sample_mask)]   # (21970, 2080)
    expr_pc_sym  = expr_pc[symbol_mask, :]                   # (20182, 2080)
    expressed    = (expr_pc_sym > 1).mean(axis=1) >= 0.10   # per-gene fraction
    combined_mask = expressed                                 # (20182,) bool

    print(f"Final gene count: {combined_mask.sum()}")   # expect 18907

    # final expression matrix
    expr_final = expr_pc_sym[combined_mask, :]   # (18907, 2080)

    # final gene arrays
    ensembl_final = ensembl[pc_mask][symbol_mask][combined_mask]
    symbol_final  = symbol_pc[symbol_mask][combined_mask]

    # --- metadata (same as before) ---
    tissue_decoded    = np.array([decode(v) for v in tissue_raw])
    tissue_harmonized = np.array([TISSUE_MAP.get(v, v) for v in tissue_decoded])
    unique, counts    = np.unique(tissue_harmonized, return_counts=True)
    count_dict        = dict(zip(unique, counts))
    tissue_final      = np.array([
        t if (count_dict.get(t, 0) >= MIN_TISSUE_COUNT and t != "Unknown") else "Other"
        for t in tissue_harmonized
    ])

    euth_decoded    = np.array([decode(v) for v in euth_raw])
    euth_harmonized = np.array([EUTHANASIA_MAP.get(v, "Unknown") for v in euth_decoded])
    euth_mapped     = euth_harmonized[sample_mask]

    strain_mapped = np.array([STRAIN_MAP.get(decode(v), decode(v))
                               for v in strain_raw[sample_mask]])
    sex_mapped    = np.array([SEX_MAP.get(decode(v), "Unknown")
                               for v in sex_raw[sample_mask]])
    study_mapped  = np.array([decode(v) for v in study_raw[sample_mask]])
    tissue_mapped = tissue_final[sample_mask]

    # --- Write ---
    with h5py.File("subset_final.h5", "w") as fout:
        fout.create_dataset("data/expression",       data=expr_final)
        fout.create_dataset("meta/genes/ensembl_id", data=ensembl_final.astype("S32"))
        fout.create_dataset("meta/genes/symbol",     data=symbol_final.astype("S32"))
        fout.create_dataset("meta/samples/spaceflight", data=flight_labels[sample_mask].astype(np.int8))
        fout.create_dataset("meta/samples/strain",   data=strain_mapped.astype("S32"))
        fout.create_dataset("meta/samples/sex",      data=sex_mapped.astype("S16"))
        fout.create_dataset("meta/samples/study_id", data=study_mapped.astype("S32"))
        fout.create_dataset("meta/samples/tissue",   data=tissue_mapped.astype("S64"))
        fout.create_dataset("meta/samples/euthanasia", data=euth_mapped.astype("S32"))


    print("=== subset_final.h5 rebuilt ===")
    print(f"Expression:  {expr_final.shape}")
    print(f"Spaceflight: {(flight_labels[sample_mask]==1).sum()} / {(flight_labels[sample_mask]==0).sum()}")
    print(f"Tissues:     {len(np.unique(tissue_mapped))}")
    print(f"Studies:     {len(np.unique(study_mapped))}")
