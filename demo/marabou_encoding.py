from maraboupy import Marabou
import numpy as np

class marabouEncoding():
	def __init__(self):
		self.var = {}
		self.netWeightMatrices = None
		self.netBiases = None
		self.numLayers = None
		self.inpDimension = None


	def checkProperties(self, autoencoder, prop, boundingBox, folder):
		outputName = 'output'
		network = Marabou.read_onnx(folder, outputName = outputName)
		inputVars = network.inputVars[0]
		outputVars = network.outputVars
		
		print(inputVars)
		print(outputVars)

		network.setLowerBound(inputVars[0],-1.0)
		network.setUpperBound(inputVars[0], 1.0)
		network.setLowerBound(inputVars[1],-1.0)
		network.setUpperBound(inputVars[1], 1.0)

		print(network.upperBoundExists(inputVars[0]))
		
		network.setLowerBound(outputVars[1], -100.0)
		network.setUpperBound(outputVars[1], 200.0)

		vals, stats = network.solve()

		print(stats.getTotalTime())
		print(stats.getNumSplits())

