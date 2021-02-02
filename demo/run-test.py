import pandas as pd
import copy as cp
import numpy as np
from autoencoder import *
from encoding import smtEncoding
import matplotlib
import matplotlib.pyplot as plt
from data_generator import *
from marabou_encoding import marabouEncoding

def main():
	
	'''
	Trains an autoencoder on (generated) data and checks adversarial robustness
	'''
	
	architecture = [10,5,10] # Change the architecture of the autoencoder according to requirement
	


	print('----------Training autoencoder----------')
	aut = autoencoder(architecture=architecture)
	data = pd.read_csv('datasets/sine_curve.csv', header=None)
	
	aut.train(data, epochs=20, learning_rate=0.01)
	
	if not aut.saveflag:
		aut.saveAE()
		print("Saving the autoencoder after training")
	

	#plot_output([data, aut.predict(data)], ['Original', 'Reconstructed'])	
	

	print("------Checking properties of autoencoders-------")


	# Parameters that can be modified
	boundingBox = 1 # Region around origin where the properties need to checked
	prop1 = ['adversarial-example', 0.1]
	prop2 = ['adversarial-robustness', [1]*10,  0.1, 0.1]
	prop3 = ['fairness', 1, 0.1]

	enc = smtEncoding()
	counterExample = enc.checkProperties(autoencoder=aut, prop=prop2, boundingBox=1)

	# For marabou
	mara = marabouEncoding()
	mara.checkProperties(autoencoder=aut, prop=prop2, boundingBox=1, folder = "Demo-aut/autoencoder.onnx")
	

	if counterExample == None:
		print("Autoencoder satisfies property is the given region")
	else:
		print("Autoencoder does not satisfy property in the given region for", counterExample)

main()
