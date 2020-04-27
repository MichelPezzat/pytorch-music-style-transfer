import numpy as np 
import os
from glob import glob

original_path = './data/test/rock/*.npy'

new_path = './data/test_new/rock'

if not os.path.exists(new_path):
		os.mkdir(new_path)

data= glob(original_path)
#print(data)


data_list = list(zip(data))
#print(data_list)

for ph in data_list:
	f = ph[0].rsplit('\\', maxsplit=1)[1]
	f_name = f.split('\\')[0]
	f = np.load(ph[0])
	print(f.shape)
	nf = f.reshape((-1,f.shape[0],f.shape[1]))
	np.save(os.path.join(new_path,f_name),nf)