from sklearn import datasets
import cv2
import matplotlib.pyplot as plt
import neunet
import numpy as np


def paintImage(image,windowtitle="",imagetitle="",axis=False):
	fig = plt.figure()
	fig.canvas.set_window_title(windowtitle)
	plt.imshow(image),plt.title(imagetitle)
	if(not axis):
		plt.xticks([]),plt.yticks([])
	plt.show()


#Probar que puedo acceder a los datos.
#for i in xrange(0,len(data),500):
#	rsdigit=data[i].reshape((8,8))
#	print("Este ejemplo es un",target[i])
#	paintImage(cv2.merge([rsdigit,rsdigit,rsdigit]))



if __name__=="__main__":
	# # PRUEBA DE LA RED NEURONAL
	#digits = datasets.load_digits()
	#data=digits["data"]
	#target=digits["target"]

	data=[]
	target=[]
	L = open("./airfoil_self_noise.dat", "r").read().splitlines();
	for line in L: 
		ex=[float(feature) for feature in line.split("\t")]
		data.append(ex[0:5])
		target.append([ex[-1]])

	data=np.float32(data)
	target=np.float32(target)

	normalized_data=np.transpose(data)
	nd=[]
	for fila in normalized_data:
		minimo=np.min(fila)
		maximo=np.max(fila)
		nd.append([((dato - minimo) / (maximo - minimo)) for dato in fila])
	normalized_data=np.transpose(np.float32(nd))


	normalized_target=np.transpose(target)
	nd=[]
	for fila in normalized_target:
		minimo=np.min(fila)
		maximo=np.max(fila)
		nd.append([((dato - minimo) / (maximo - minimo)) for dato in fila])
	normalized_target=np.transpose(np.float32(nd))




	layers=[5,4,4,1]
	nn=neunet.NeuronalNet(normalized_data,normalized_target,0.1,layers,random_weights=True)
	nn.describeNet()
	d_tr,l_tr,d_ts,l_ts=nn.getTrainingAndTest(1,0)
	print("Entrenando la red...")
	nn.trainNet(d_tr,l_tr,20)
	print("Red entrenada.")

	
	

	result1=nn.predict([ 0.03030303,  0.,          1.,          1.,          0.03900472]) #Deberia salir [ 0.60682845]
	result2=nn.predict([ 0.00191919,  0.3,          0.2,          0.7,          0.33900472]) #Deberia salir [ ???]
	print(result1)
	print(result2)

	