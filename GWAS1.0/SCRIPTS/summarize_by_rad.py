import json
import pandas as pd
import sys

all_rads=dict()
json_file=sys.argv[1]

with open(json_file, 'r') as f:
    results=json.load(f)
    for chrom in results:
        for rads in results[chrom]:
            for rad in rads.keys():
                if not rad in all_rads:
                    all_rads[rad] = 0
                all_rads[rad] += len(rads[rad])
f.close()

print(all_rads)
