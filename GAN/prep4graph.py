#gamma: 0.98 ld: 8 bs: 12 nl: 2 hd: 128
import pandas
import sys
import matplotlib.pyplot as plt


inputFile=open(sys.argv[1], 'r')
lines = inputFile.readlines()
outputDict=dict()
outputDict['ld'] = list()
outputDict['hd'] = list()
outputDict['bs'] = list()
outputDict['nl'] = list()
outputDict['lr'] = list()
for line in lines:
	lineArray = line.split(':')
	# ['gamma', ' 0.96 ld', ' 32 bs', ' 16 nl', ' 3 hd', ' 256 lr', '1e-04\n']
	gamma = lineArray[1].split('ld')[0].strip()
	ld = lineArray[2].split('bs')[0].strip()
	bs = lineArray[3].split('nl')[0].strip()
	nl = lineArray[4].split('hd')[0].strip()
	hd = lineArray[5].split('lr')[0].strip()
	lr = lineArray[6].strip()
	outputDict['ld'].append((float(ld), float(gamma)))
	outputDict['bs'].append((float(bs), float(gamma)))
	outputDict['nl'].append((float(nl), float(gamma)))
	outputDict['hd'].append((float(hd), float(gamma)))
	outputDict['lr'].append((float(lr), float(gamma)))


print(outputDict)
		
