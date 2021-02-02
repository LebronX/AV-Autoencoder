from autoencoder import *
from z3 import *
import time

class smtEncoding():
	def __init__(self):
		self.var = {}
		self.netWeightMatrices = None
		self.netBiases = None
		self.numLayers = None
		self.inpDimension = None
		self.solver = Solver()


	def checkProperties(self, autoencoder, prop, boundingBox):
		'''
		checks if the prop holds, returns a counterexample otherwise
		'''


		self.numLayers = autoencoder.module.numLayers
		self.inpDimension = autoencoder.module.inputDimension 
		self.constructAEMatrixBias(autoencoder)


		if prop[0] == 'adversarial-example':
			self.AEConstr(autoencoder)
			self.advExampleConstr(epsilon=prop[1])
			self.boundingBoxConstr(boundingBox)

		if prop[0] == 'adversarial-robustness':
			self.AEConstr(autoencoder)
			inp_torch = torch.Tensor(prop[1])
			out_point = autoencoder.module(inp_torch)[1].tolist()


			self.advRobustnessConstr(inp_point=prop[1], out_point=out_point, delta=prop[2], epsilon=prop[3])
			self.boundingBoxConstr(boundingBox)

		


		if prop[0] == 'fairness':
			self.AEConstr(autoencoder, varName='x')
			self.AEConstr(autoencoder, varName='y')
			self.fairnessConstr(feature=prop[1],epsilon=prop[2])
			self.boundingBoxConstr(boundingBox, varName='x')
			self.boundingBoxConstr(boundingBox, varName='y')

		



		result = self.solver.check()
		if  result == sat:
			model = self.solver.model()		
			
			if prop[0] == 'fairness':
				point1 = self.modelToPoint(model, 'x_0')
				point2 = self.modelToPoint(model, 'y_0')
				example = (point1, point2)
			else:
				example = self.modelToPoint(model, 'x_0')
		else:
			example = None

		return example
		

	def constructAEMatrixBias(self, autoencoder):
		'''
		constructs the autoencoder network weight matrices 
		'''
		self.netWeightMatrices = []
		self.netBiases = []
		localCount = 1
		for param in autoencoder.module.parameters():
	
			if localCount == 1:
				self.netWeightMatrices.append(np.random.normal(size = (param.size()[1], param.size()[1])))
				localCount = 2
			if len(param.size()) >= 2:
				self.netWeightMatrices.append(param.detach().numpy())
				localCount+=1
			else:
				self.netBiases.append(param.detach().numpy())




	def AEConstr(self, autoencoder, varName='x'):
		'''
		adds the constraints for autoencoder 
		'''
		
		self.var[varName] = []
		for layer in range(self.numLayers):
			self.var[varName].append([Real(varName+'_' +str(layer)+'_'+str(i)) for i in range(self.netWeightMatrices[layer].shape[0])])

	
		constr = []
		for layer in range(1,self.numLayers-1):
			for destNeuron in range(len(self.var[varName][layer])):
			
				weightedSum = Sum([self.netWeightMatrices[layer][destNeuron][sourceNeuron] * self.var[varName][layer-1][sourceNeuron] 
								for sourceNeuron in range(len(self.var[varName][layer-1]))])
				
				constr.append(self.var[varName][layer][destNeuron] == 
							If(weightedSum + self.netBiases[layer-1][destNeuron] < 0, 0, 
								weightedSum + self.netBiases[layer-1][destNeuron]))

		# Output layer
		for destNeuron in range(len(self.var[varName][self.numLayers-1])):

			weightedSum = Sum([self.netWeightMatrices[self.numLayers-1][destNeuron][sourceNeuron] * self.var[varName][self.numLayers-1-1][sourceNeuron] 
							for sourceNeuron in range(len(self.var[varName][self.numLayers-1-1]))])
			
			constr.append(self.var[varName][self.numLayers-1][destNeuron] == weightedSum + self.netBiases[self.numLayers-2][destNeuron])

		self.solver.add(constr)


	

	def advExampleConstr(self, epsilon, varName='x'):
		'''
		adds constraints for adversarial examples
		'''
		

		#l-infinity distance
		# var[varName][0] is the input layer and var[varName][self.numLayers-1] is the output layer
		
		constr=Or([(Or(self.var[varName][0][i] - self.var[varName][self.numLayers-1][i] > epsilon, 
							self.var[varName][self.numLayers-1][i] - self.var[varName][0][i] > epsilon))
						for i in range(self.inpDimension)])

		self.solver.add(constr)



	def advRobustnessConstr(self, inp_point, out_point, delta, epsilon, varName = 'x'):
		'''
		generates constraints for checking adversrial robutness
		'''
		conjuct1 = And(
					[And(inp_point[i] - self.var[varName][0][i] < delta, 
							self.var[varName][0][i] - inp_point[i] < delta)
							for i in range(self.inpDimension)])
				
		conjuct2 = Or(
					[Or(out_point[i] - self.var[varName][self.numLayers-1][i] > epsilon, 
							self.var[varName][self.numLayers-1][i] - out_point[i] > epsilon)
							for i in range(self.inpDimension)]
					)

		constr = And(conjuct1, conjuct2)

		self.solver.add(constr)





	def fairnessConstr(self, feature, epsilon, varName1='x', varName2='y'):


		assert(feature<=self.inpDimension)
		
		conjuct1 = And( Not(self.var[varName1][0][feature-1] == self.var[varName2][0][feature-1]),
						And([self.var[varName1][0][i] == self.var[varName1][0][i]
							for i in range(self.inpDimension) if i != feature-1]))
				
		conjuct2 = Or(
					[Or(self.var[varName1][self.numLayers-1][i] - self.var[varName2][self.numLayers-1][i] > epsilon, 
							self.var[varName2][self.numLayers-1][i] - self.var[varName2][self.numLayers-1][i] > epsilon)
							for i in range(self.inpDimension)]
					)

		constr = And(conjuct1, conjuct2)

		self.solver.add(constr)


	def boundingBoxConstr(self, boundingBox, varName='x'):

		constrs = And([And(self.var[varName][0][i] < boundingBox, self.var[varName][0][i] > -boundingBox) 
														for i in range(self.inpDimension)])
		self.solver.add(constrs)


	def modelToPoint(self, model, variable):
		'''
		converts the model to a point
		'''		
		sortedModel = sorted([(var, model[var]) for var in model], key = lambda x: str(x[0]))
		
		point = []
		for elem in sortedModel:
			if variable in str(elem[0]):
				numerator = elem[1].numerator_as_long()
				denominator = elem[1].denominator_as_long()
				decimal = float(numerator/denominator)
				point.append(decimal)
		return point

		



	
	def clearSmt(self):

		self.var = {}
		self.netWeightMatrices = None
		self.netBiases = None
		self.satisfiable = None
		self.solver = Solver()

