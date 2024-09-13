import io
import torch
import sys
from fl_src.irm_module import IRMModule, NLIRMModule
from fl_src.erm_module import ERMModule
import argparse

output_data_regime="real-valued"
num_classes=1
metric='MSE'
seed=0

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-pf', '--protobufFile', help='protobuf file name', default=None, required=True)
	parser.add_argument('-tf', '--trainFile', help='training data file name', default=None, required=True)
	parser.add_argument('-nf', '--numFeatures', help='number of features', default=None, required=True)

	parser.add_argument('-mn', '--modelName', help='name of model', default=None, required=True)
	parser.add_argument('-pw', '--printWeights', help='print weights of model', default='False', required=False)
	
	return parser.parse_args()

def main():
	args=parse_args()
	inputProtoFile = args.protobufFile
	modelType=args.modelName
	printWeights = eval(args.printWeights)
	numFeatures = int(args.numFeatures)
	with open(inputProtoFile, 'rb') as f:
		en_model=f.read()
	f.close()
	trainFile = args.trainFile
	with open(trainFile, 'r') as f:
		header=f.readline().strip()
	print(header)

	
	weights = torch.load(io.BytesIO(en_model), map_location=torch.device('cpu'))["model_state_dict"]
	if printWeights:
		print(weights)

	for key, value in weights.items():
		#print('key: ', str(key))
		#print('value: ', str(value))
		prefix = key.split("phi")[0]
		in_size = value.shape[1]
		break


	if modelType == 'IRM':
		model = IRMModule(logger=None, prefix='nlirm_')
	elif modelType == 'NLIRM':
		model = NLIRMModule(logger=None, prefix='nlirm_')
	elif modelType == 'ERM':
		model = ERMModule(logger=None, prefix='nlerm_')
	model.init_network(in_size, num_classes, output_data_regime=output_data_regime, seed=seed)
	results=model.results()

	coefficients = results['to_bucket']['coefficients']

	featureDict = dict()
	features = header.split(',')
	for i in range(numFeatures): 
		feature = features[i]
		# tensor([-0.1322], grad_fn=<SelectBackward0>) 
		featureDict[feature] = coefficients[i].item()

	print(featureDict)	

if __name__ == '__main__':
	main()
