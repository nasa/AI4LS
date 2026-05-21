import gseapy as gp
libs = gp.get_library_name(organism="mouse")
for l in sorted(libs):
    if any(k in l.lower() for k in ["kegg", "wiki", "reactome", "go_bio"]):
        print(l)
