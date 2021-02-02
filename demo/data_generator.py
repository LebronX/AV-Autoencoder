import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import *
import pandas as pd


def generate_straight_line(length=1000, freq=0.01,	slope = 2, const = 1, noise_variance=0.1):
	'''
	generates a noisy straight line
	'''
	x_values = np.array([i*freq for i in range(length)])
	y_values = np.array([i*slope + const + np.random.normal(0,noise_variance) for i in x_values])


	y_values = pd.DataFrame(y_values.reshape((1,len(y_values))))
	y_values.to_csv("datasets/straight_line.csv", index=False, header=False, sep=',')

	plot_output(datasets=[y_values], plotnames=["-straight_line"], freq=freq, savefile='straight_line.png')


def generate_sine_curve(length=200, freq=0.1, scale=1, noise_variance=0.05):
	'''
	generates a noisy sine curve
	'''
	x_values = np.array([i*freq for i in range(length)])
	y_values = np.array([scale*np.sin(i) + np.random.normal(0,noise_variance) for i in x_values])


	y_values = pd.DataFrame(y_values.reshape((1,len(y_values))))
	y_values.to_csv("datasets/sine_curve.csv", index=False, header=False, sep=',')

	plot_output(datasets=[y_values], plotnames=["-sine_curve"], freq=freq, savefile='sine_curve.png')




def plot_output(datasets, plotnames, freq=0.01, savefile=None):
	'''
	plots a (csv) dataset 
	'''
	for k in range(len(datasets)):
		dataset = datasets[k]
		record_num = 0
		for i in dataset.iterrows():
			record_num+=1  
			y_values = i[1].to_numpy()
			x_values = np.array([i*freq for i in range(len(y_values))])

			if y_values.shape != x_values.shape:
				raise Exception("X axis and Y axis do not have equal datapoints")

			plt.plot(x_values, y_values, label="plot"+plotnames[k]+str(record_num))
			plt.legend()

	if savefile != None:
		plt.savefig(savefile)

	plt.show()

