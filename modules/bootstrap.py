import random
import numpy as np
def bootstrap(data,dl,dm):
	
	N_1 = 250 # number used for training sample generation
	N_2 = 150 # number used for testing sample generation
	n_train = 250 # number of training samples to be generated
	n_test = 150 # number of testing samples to be generated
	data_train = np.zeros(n_train,2,640)
	data_test = np.zeros(n_test,2,640)
	idx = np.arange(0,N_1+N_2) # N_1 + N_2 is total number of the samples we have for each damage scenerio
	random.shuffle(idx) # randomly shuffling the samples
	idx_train = idx[0:N_1] #Dividing the dataset into N_1 and N_2 dataset
	idx_test = idx[N_1:end]
	X=100 # number of samples used to calculate the average
	DL  = dl[0] # damage location of the damage scenerio
	DM  = dm[0] # damage magnitude of the damage scenerio	
	for i in range(n_train):
		random.shuffle(idx_train)
		id = idx_train[0:X]
		d = data[id,:,:]
		d_avg = np.mean(d,axis=0)
		data_train[i,:,:] = d_avg
	for i in range(n_test):
		random.shuffle(idx_test)
		id = idx_test[0:X]
		d = data[id,:,:]
		d_avg = np.mean(d,axis=0)
		data_test[i,:,:] = d_avg
			
return data_train,data_test,DL,DM
