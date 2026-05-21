import requests
r = requests.get("https://genelab-data.ndc.nasa.gov/genelab/data/study/data/OSD-104/")
print(r.json())
