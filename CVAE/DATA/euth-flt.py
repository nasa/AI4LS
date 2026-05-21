import h5py
import numpy as np

def decode(val):
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8").strip()
    return str(val).strip()

with h5py.File("OSDR_mouse_RNAseq_Feb2026.h5", "r") as f:
    euth_raw    = f["meta"]["samples"]["parameters"]["study.parameter value.euthanasia method"][:]

    flight_raw  = f["meta"]["samples"]["factors"]["study.factor value.spaceflight"][:]
    strain_raw  = f["meta"]["samples"]["characteristics"]["study.characteristics.strain"][:]
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


flight_labels  = np.array([FLIGHT_MAP.get(decode(v), None) for v in flight_raw])
strain_decoded = np.array([decode(v) for v in strain_raw])
sample_mask    = np.array([
    flight_labels[i] is not None and strain_decoded[i] != ""
    for i in range(len(flight_labels))
])

euth_decoded    = np.array([decode(v) for v in euth_raw[sample_mask]])
euth_harmonized = np.array([EUTHANASIA_MAP.get(v, "Unknown") for v in euth_decoded])

unique, counts = np.unique(euth_harmonized, return_counts=True)
print("Euthanasia method distribution in filtered dataset:")
for c, u in sorted(zip(counts, unique), reverse=True):
    print(f"  {int(c):4d}  {u}")
