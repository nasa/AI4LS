import matplotlib.pyplot as plt
import ast
import numpy
import statistics as s
import sys

inputFile=sys.argv[1]
with open(inputFile, 'r') as f:
    data = f.read()
f.close()
d = ast.literal_eval(data)
pDict = dict()
aDict = dict()
for k in d:
    aDict[k] = dict()
    pDict[k] = dict()
    for p in d[k]:
        x=str(p[0])
        if not x in pDict[k]:
            pDict[k][x] = list()
        pDict[k][x].append(p[1])
    for v in pDict[k]:
        aDict[k][v] = s.mean(pDict[k][v])
    print(k, aDict[k])
    # scatter
    x = list(map(float, list(aDict[k].keys())))
    y = list(map(float, list(aDict[k].values())))
    plt.figure()
    plt.scatter(x, y)
    plt.title(k)
    #plt.show()
    plt.savefig(k + '-perf-scatter.png')

#plt.scatter(x[0:1000, 0], x[0:1000, 1], c='k', marker='^', zorder=10, edgecolors='none', s=10)
#plt.scatter(x[1000:, 0], x[1000:, 1], c='w', marker='o', zorder=10, edgecolors='none', s=10)