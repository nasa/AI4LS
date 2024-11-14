import matplotlib.pyplot as plt
import sys
import pandas as pd

exprFile=sys.argv[1]
expr=pd.read_csv(exprFile, sep=',', header=0)

gene=sys.argv[2]
geneExpr = expr[expr['gene'] == gene]




count, bins, ignored = plt.hist(geneExpr, 30, density=True)
plt.show()