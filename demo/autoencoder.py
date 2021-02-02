import math
import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import copy
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt




class autoencoder:
	def __init__(
		self,
		architecture,
		name='Demo-aut',
		activationFct = nn.ReLU(),
		initialization = nn.init.xavier_normal_):

		self.name = name
		self.activationFct = activationFct
		self.initialization = initialization
		self.architecture = architecture
		self.module = autoencoderModule(self.architecture, self.activationFct, self.initialization)
		self.saveflag = 0

	def train(self, data: pd.DataFrame, epochs=50, batch_size=20, learning_rate = 0.01, timeSeries=True):
		'''
		trains an autoencoder on given data
		'''
		data = data.values.astype(np.float32)
		dataTorch = torch.tensor(data)

		if timeSeries:
			
			
			splits = self.architecture[0]
			count=0
			for record in dataTorch:

				for j in range(len(record)-splits+1):

					
					current_split = torch.reshape(record[j:(j+splits)].clone().detach(), (1,splits))
					if count==0:
					
						record_split = current_split
						count=1
					else:
						record_split = torch.cat((record_split,current_split), 0)
						

		dataTorch = record_split	
		
		
		optimizer = optim.Adam(self.module.parameters(), lr = learning_rate)
		criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
		for j in range(epochs):
			epochLoss = 0
			for records in DataLoader(dataTorch, batch_size= batch_size):
				inputs = records
				optimizer.zero_grad()
				outputs = self.module(inputs)[1]
				loss = criterion(inputs, outputs)
				epochLoss = epochLoss + loss
				loss.backward()
				optimizer.step()

			if float(loss)<0.01:
				print("Saving the current model after epoch %d"%j)
				self.saveAE()
				self.saveflag = 1
		


	def predict(self, data: pd.DataFrame, timeSeries=True):
		'''
		outputs the result of applying autoencoder on data
		'''
		data = data.values.astype(np.float32)
		
		dataPredictedDf = pd.DataFrame()
		for record in DataLoader(data):

			if timeSeries:
				dataPredicted = []
				splits = self.architecture[0]
				split_predictions = []
				record1 = record.squeeze()				
				numSplits = len(record1)-splits+1
				for j in range(numSplits):

					current_split = torch.reshape(record1[j:(j+splits)].clone().detach(), (1,splits))
					current_prediction = self.module(current_split)[1].detach().numpy().squeeze()

					split_predictions.append(current_prediction)
				
				for j in range(len(record1)): 

					l = [split_predictions[i][j-i] for i in range(min(j+1,numSplits)) if j-i<splits]
					dataPredicted.append(sum(l)/len(l))

				dataPredicted = np.array(dataPredicted)

				dataPredictedDf = pd.DataFrame.append(dataPredictedDf,pd.Series(dataPredicted), ignore_index=True)

			else:
				dataPredicted = np.append(dataPredicted, self.module(record)[1].detach().numpy())
				dataPredictedDf = pd.DataFrame.append(dataPredicted)
			
		return dataPredictedDf

	def saveAE(self, folder=None):
		'''
		stores an autoencoder in the folder given by 'folder'.
		'''

		if folder==None:
			folder = self.name

		cwd = os.getcwd()
		if not os.path.exists(folder):
 			os.makedirs(folder)
		os.chdir(folder)
		

		torch.save(self.module.state_dict(), 'autoencoder.pt')
		algorithmDictAdj = copy.deepcopy(self.__dict__)
		
		# .onnx file for marabou
		dummy_input = torch.randn(10)
		torch.onnx.export(self.module, dummy_input, "autoencoder.onnx", output_names = ['output'])	

		for key in list(algorithmDictAdj.keys()):
			algorithmDictAdj[key] = str(algorithmDictAdj[key])

		with open('parameters.txt', 'w') as jsonFile:
			json.dump(algorithmDictAdj, jsonFile, indent = 0)

		os.chdir(cwd)


	def loadAE(self, folder=None):
		'''
		loads an autoencoder saved in the folder 'folder' 
		'''
		if folder==None:
			folder = self.name
			if not os.path.exists(folder):
				raise Exception("Enter a valid folder name")

		self.module.load_state_dict(torch.load(folder+'/autoencoder.pt'))



class autoencoderModule(nn.Module):
	def __init__(self, architecture, activationFct, initialization):
		super(autoencoderModule, self).__init__()
		self.architecture = architecture
		self.activationFct = activationFct
		self.initialization = initialization
		self.numLayers = len(self.architecture)
		self.inputDimension = self.architecture[0]
		bottleneck = np.argmin(self.architecture)
		
		if self.architecture[0] != self.architecture[-1] or \
			self.architecture[0]<=self.architecture[bottleneck]:
			
			raise Exception("Not an autoencoder architecture")

		encLayers = []
		decLayers = []
		for layer in range(self.numLayers-1):

			if layer<bottleneck:

				encLayers.append(nn.Linear(self.architecture[layer], self.architecture[layer+1]))
				encLayers.append(self.activationFct)


			elif layer >= bottleneck and layer<(self.numLayers-1-1):
				decLayers.append(nn.Linear(self.architecture[layer], self.architecture[layer+1]))
				decLayers.append(self.activationFct)
			else:

				decLayers.append(nn.Linear(self.architecture[layer], self.architecture[layer+1]))


		self.encoder = nn.Sequential(*encLayers)
		self.decoder = nn.Sequential(*decLayers)
		
		for layer in self.encoder:
			if isinstance(layer, nn.Linear):
				self.initialization(layer.weight)
		for layer in self.decoder:
			if isinstance(layer, nn.Linear):
				self.initialization(layer.weight)


	def forward(self, inputData):
		encoding = self.encoder(inputData)
		decoding = self.decoder(encoding)
		return (encoding, decoding)


