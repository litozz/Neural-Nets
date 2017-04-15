#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import math
from random import shuffle

def sigmoid(x):
	return 1/(1+math.e**(-x)) 	#Sigmoid

def sigmoid_diff(x):
	return (1/((1+math.e**(-x))**2))*(math.e**(-x)) 	#Sigmoid diff


class InputNeuron:
	def __init__(self):
		self.n_inputs=1
		self.inputs=[]
		self.weights=[1]
	
	def getInputs(self):
		return self.inputs

	def getNInputs(self):
		return 1

	def setInputs(self,_inputs):
		self.inputs=_inputs

	def getWeights(self):
		return self.weights
	
	def computeExit(self):
		#print("INITIAL-INPUTS: {}".format(self.inputs))
		#print("INITIAL-PESOS: {}".format(self.weights))
		return self.inputs

	def getActivationThresold(self):
		return 1






class Neuron:
	def __init__(self,_n_inputs,_activation_thr):
		self.act_thr=_activation_thr
		#self.inputs=np.float32(_inputs)
		#self.weights=np.float32(_weights)
		self.n_inputs=_n_inputs
		self.inputs=[]
		self.weights=[]

	def activation_func(self,x):
		return sigmoid(x)

	def activation_func_diff(self,x):
		return sigmoid_diff(x)

	def getNInputs(self):
		return self.n_inputs
	
	def getInputs(self):
		return self.inputs

	def getWeights(self):
		return self.weights

	def getActivationThresold(self):
		return self.act_thr

	def setInputs(self,_inputs):
		if(len(_inputs)==self.n_inputs):
			self.inputs=np.float32(_inputs)

	def setWeights(self,_weights):
		if(len(_weights)==self.n_inputs):
			self.weights=np.float32(_weights)

	def setActivationThresold(self,_activation_thr):
		self.act_thr=_activation_thr

	def computeExit(self):
		#print("INPUTS: {}".format(self.inputs))
		#print("PESOS: {}".format(self.weights))

		npesos=len(self.weights)
		if( npesos == len(self.inputs) and npesos>0 ):
			return self.activation_func(self.act_thr+sum(self.weights*self.inputs))
		else:
			raise IndexError, "Inputs must have the same length of weights and at least 1"

	#Derivada de la activacion con respecto al umbral de activacion (SOLO FUNCIONA CON LA SIGMOIDE)
	def computeDiffA_respectThr(self):
		a=self.computeExit()
		return a*(1-a)

	#Derivada de la activacion con respecto a un determinado peso  (SOLO FUNCIONA CON LA SIGMOIDE)
	def computeDiffA_respectWeight(self,i):
		a=self.computeExit()
		return a*(1-a)*self.inputs[i]
	










class NeuronalNet:
	def __init__(self,_data,_labels, _learning_factor,_vector_layers,random_weights=True):
		nfeatures=_data.shape[1]
		try:
			noutputs=_labels.shape[1]
		except:
			noutputs=1
			_labels=np.float32([[label] for label in _labels])

		if(nfeatures==_vector_layers[0] and noutputs==_vector_layers[-1] and len(_data)==len(_labels)): #El numero de entradas y el numero de salidas son iguales
			self.data=_data
			self.labels=_labels
			print(self.labels)
			self.learning_factor=_learning_factor
			

			self.n_input=_vector_layers[0]
			self.n_total_layers=len(_vector_layers)
			self.hidden_layers=[i for i in _vector_layers[1:self.n_total_layers-1] ]
			self.n_hidden_layers=len(self.hidden_layers)
			self.n_output=_vector_layers[-1]

			self.neurons_per_layer=_vector_layers

			self.weights=[]
			if(random_weights):
				for i in range(self.n_total_layers-1):
					sinapsis_layer=[]
					for j in range(_vector_layers[i]):
						sinapsis_layer.append([random.uniform(-1, 1) for neuron in range(_vector_layers[i+1])])
					sinapsis_layer=np.float32(sinapsis_layer)
					self.weights.append(sinapsis_layer)
			else:
				for i in range(self.n_total_layers-1):
					sinapsis_layer=[]
					for j in range(_vector_layers[i]):
						sinapsis_layer.append([0 for neuron in range(_vector_layers[i+1])])
					self.weights.append(sinapsis_layer)

			self.neuronNet=[]
			activation_thr=0.5

			input_layer=[InputNeuron() for i in range(self.n_input)] #Creo la capa de entrada de neuronas
			self.neuronNet.append(input_layer)

			for i in range(1,len(_vector_layers)):
				h_layer=[]
				#La traspuesta me dice como se relaciona la neurona de la capa siguiente
				#con las de la capa anterior
				sinap_layer=np.transpose(self.weights[i-1])

				for j in range(_vector_layers[i]):
					neuron=Neuron(_vector_layers[i-1],activation_thr)
					neuron.setWeights(sinap_layer[j])
					h_layer.append( neuron )
				self.neuronNet.append(h_layer)
		else:
			raise IndexError, """Input layer must be the same length of number of features.
			Output layer must be the same length of number of label outputs."""




	def getExample(self,i):
		return (self.data[i],self.labels[i])

	def getWeight(self,sinapsis_layer,first_neuron,second_neuron):
		return self.weights[sinapsis_layer][first_neuron][second_neuron]

	def getNInputs(self):
		return self.n_input

	def getNHiddenLayers(self):
		return self.n_hidden_layers

	def getNNeuronsHiddenLayer(self,i):
		return self.hidden_layers[i]

	def getNOutputs(self):
		return self.n_output

	def getSinapsisLayer(self,i):
		return self.weights[i]


	
	#####################################
	#SOLO PARA QUE FUNCIONE				#
	#def diff_f(self,x):				#
	#	return 2*(x-1)					#
	#####################################
	

	def printWeights(self):
		print("[")
		for i in range(len(self.weights)):
			print("\t[ Capa de sinapsis {}".format(i))
			for j in range(len(self.weights[i])):
				print("\t\t{}".format(self.weights[i][j]))
			print("\t]")
		print("]")

	def printNeuronalNet(self):
		print("[")
		for i in range(len(self.neuronNet)):
			print("\t["),
			for j in range(len(self.neuronNet[i])):
				if(i==0):
					print("IN_X{}({})\t".format(j,self.neuronNet[i][j].getNInputs())),
				else:
					print("N_{}_{}({})\t".format(i,j,self.neuronNet[i][j].getNInputs())),
			print("]")
			print("")
		print("]")

	def printNeuron(self,layer,i):
		neuron=self.neuronNet[layer][i]
		print("----------------------------------------------------")
		print("Numero de entradas: {}".format(neuron.getNInputs()))
		
		inputs=neuron.getInputs()
		print("Entradas:")
		for i in range(len(inputs)):
			print("\tEntrada {}: {}".format(i,inputs[i]))

		weights=neuron.getWeights()
		print("Pesos sinapticos:")
		for i in range(len(weights)):
			print("\tEntrada {}: {}".format(i,weights[i]))

		output=neuron.computeExit()
		print("Salida: {}".format(output))
		print("----------------------------------------------------")



	def describeNet(self):
		print("Descripcion de la red neuronal:")
		print("-------------------------------")
		print("* Numero de entradas: {}".format(self.n_input))
		n_hidden=len(self.hidden_layers)
		print("* Numero de capas ocultas: {}".format(n_hidden))
		for i in range(n_hidden):
			print("\t - Capa oculta {}: {} neuronas".format(i+1,self.hidden_layers[i]))
		print("* Numero de salidas: {}".format(self.n_output))
		print("* Matriz de pesos: ")
		self.printWeights()
		print("* Estructura de la red neuronal: N_capa_numNeurona(numero_inputs)")
		self.printNeuronalNet()

	def error_function(self, desired_output, obtained_output):
		#e(s,y)=1/2*sum(s_i-yi)
		return 0.5*sum([( (desired_output[i]-obtained_output[i])*(desired_output[i]-obtained_output[i]) ) for i in range(len(desired_output)) ])

	def error_function_diffRespectYi(self, desired_output, obtained_output,i): #derivada del error con respecto a ¿¿¿¿¿¿¿¿¿¿??????????
		#de/d(y_i) = -(s_i-yi)
		return -(desired_output[i]-obtained_output[i])
		
	def calculateDeltas(self,desired_output,obtained_output):
		last_neuron_layer=self.n_total_layers-1
		#Calculo los deltas de la capa de salida
		#delta=( y_i*(1-y_i)(-s_i-y_i) )
		deltas=[[self.neuronNet[last_neuron_layer][i].computeDiffA_respectThr() * self.error_function_diffRespectYi(desired_output,obtained_output,i) for i in range(self.neurons_per_layer[last_neuron_layer])]]
		#Calculo todas las deltas restantes
		for layer in range(last_neuron_layer-1,0,-1):
			deltas_per_layer=[]
			for neuron in range(self.neurons_per_layer[layer]):
				#delta=( a_i^(c)*(1-a_i^(c))*sum_i^nNeurons(c) w_i(i+1) )
				deltas_per_layer.append(self.neuronNet[layer][neuron].computeDiffA_respectThr() * sum([ self.weights[layer][neuron][neuron_sig] for neuron_sig in range(len(deltas[last_neuron_layer-(layer+1)])) ]))
			deltas.append(deltas_per_layer)

		#Deltas en la capa de entrada
		deltas.append([0 for i in range(self.neurons_per_layer[0])])
		
		deltas = [deltas[layer] for layer in range(len(deltas)-1,-1,-1)]
		return deltas

	def computeErrorDiff_respectWeight(self,deltas,j,i,c):
		#print(deltas)
		if(c>0):#Para calcular el error tenemos que estar en una capa oculta o en la de salida. No hay error en la de entrada
			# de/d( w_ji^(c-1) ) = a_j^(c-1) * delta_ic
			return self.neuronNet[c-1][j].computeExit() * deltas[c][i]

	def computeErrorDiff_respectBias(self,deltas,i,c):
		if(c>0):#Para calcular el error tenemos que estar en una capa oculta o en la de salida. No hay error en la de entrada
			# de/d( u_i^(c) ) = delta_ic
			return deltas[c][i]







	#####################################################################
	# Este metodo calcula el minimo de una funcion usando su derivada	#
	# mediante el algoritmo de gradiente desdendente					#
	#####################################################################
	#def gradientDescent(self,epsilon, start_point):
	#	start_value=self.diff_f(start_point)
	#	w_i=start_point												# w0
	#	w_sig=start_point-(self.learning_factor*start_value)		# w1=w0 - (nabla * f'(w0))
		
	#	while(abs(self.diff_f(w_i)-self.diff_f(w_sig))>epsilon):	# ¿( f'(w_i)-f'(w_i+1) ) > epsilon?
	#		w_i=w_sig												# actualizo w_i
	#		w_sig=w_i-(self.learning_factor*self.diff_f(w_i))		# wi+1=wi - (nabla * f'(wi))
			
	#	return w_sig



	def predict(self,example,label=None):
		#Introduzco el ejemplo por el input
		for i in range(len(self.neuronNet[0])):					# Por cada neurona en la capa 0 (inicial)
			self.neuronNet[0][i].setInputs(example[i])			# Introduzco en una neurona la característica correspondiente

		for layer in range(1,self.n_total_layers):										#Por cada capa K>1
			inputs=[ant_neuron.computeExit() for ant_neuron in self.neuronNet[layer-1]]	#Calculo el vector de inputs con las salidas de la capa anterior

			for neuron in self.neuronNet[layer]: 				#Por cada neurona en la capa K			
				neuron.setInputs(inputs)						#Añado el vector de inputs

		v_output=[]
		for output_neuron in self.neuronNet[self.n_total_layers-1]:
			v_output.append(output_neuron.computeExit())

		if(label==None):
			return v_output
		else:
			error=self.error_function(label,v_output)
			return (v_output,error)



	def trainNet(self,train_data,train_label, nEpoch):
		if(len(train_data)==len(train_label)):
			for epoch in range(nEpoch):
				print("Completado {}% del entrenamiento...".format((float(epoch)/float(nEpoch))*100))
				print(self.weights[0][0][0])
				error=0
				result=None
				for example,label in zip(train_data,train_label):
					result,error=self.predict(example,label)
					
					#BACKPROPAGATION
					deltas=self.calculateDeltas(label,result)
					for layer in range(self.n_total_layers-1,0,-1):			#Para cada capa desde la ultima
						for i in range(len(self.neuronNet[layer])):			#Para cada neurona en la capa actual	
							
							#Actualizacion de pesos
							new_weights=[]	
							for j in range(len(self.neuronNet[layer-1])):	#Para cada neurona en la capa anterior a la actual
								#print("Peso sinaptico entre la neurona {} en la capa {} y la neurona {} en la capa {}: {}".format(i,layer,j,layer-1,self.weights[layer-1][j][i]))

								act_weight=self.weights[layer-1][j][i]
								#print("\tPeso actual: {}".format(act_weight))
								diff_respect_weight=self.computeErrorDiff_respectWeight(deltas,j,i,layer)
								#print("\t\tLa derivada con respecto al peso vale: {}".format(diff_respect_weight))
								self.weights[layer-1][j][i] = self.weights[layer-1][j][i] - (self.learning_factor * diff_respect_weight)
								act_weight=self.weights[layer-1][j][i]
								#print("\tPeso CORREGIDO: {}".format(act_weight))
								new_weights.append(self.weights[layer-1][j][i])

							self.neuronNet[layer][i].setWeights(new_weights)

							
							#Actualizacion del umbral de activacion (bias)
							new_bias=0	
							act_bias=self.neuronNet[layer][i].getActivationThresold()
							diff_respect_bias=self.computeErrorDiff_respectBias(deltas,i,layer)
							#print("\t\tLa derivada con respecto al umbral de activacion vale: {}".format(diff_respect_bias))
							new_bias=act_bias - (self.learning_factor * diff_respect_bias)
							self.neuronNet[layer][i].setActivationThresold(new_bias)
							#if(layer!=0):	
							#	print("\tBias actual: {}".format(self.neuronNet[layer][i].getActivationThresold()))
							#	print("\tBias CORREGIDO: {}".format(self.neuronNet[layer][i].getActivationThresold()))
							#else:
							#	print("\tBias actual: {}. ESTOY EN CAPA INPUT Y NO ACTUALIZO EL UMBRAL DE ACTIVACION (BIAS) SIEMPRE DEBE SER 1.".format(self.neuronNet[layer][i].getActivationThresold()))
							#	print("\tBias CORREGIDO: {}. ESTOY EN CAPA INPUT Y NO ACTUALIZO EL UMBRAL DE ACTIVACION (BIAS) SIEMPRE DEBE SER 1.".format(self.neuronNet[layer][i].getActivationThresold()))
			
		else:
			raise IndexError, "Train set and and train label must have the same length"




	def getTrainingAndTest(self,perc_tr,perc_test,shuff=True):
		n_examples=len(self.data)
		index_list=[i for i in range(n_examples)]
		if(shuff):
			shuffle(index_list)

		
		index_tr = index_list[ 0 : int(perc_tr*n_examples) ]
		index_test=index_list[ int(perc_tr*n_examples) : -1 ]

		data_training = np.float32([self.data[i] for i in index_tr])
		label_training = np.float32([self.labels[i] for i in index_tr])

		data_test = np.float32([self.data[i] for i in index_test])
		label_test = np.float32([self.labels[i] for i in index_test])

		return data_training,label_training,data_test,label_test