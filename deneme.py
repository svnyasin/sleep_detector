import h5py
import numpy as np

filename = "models/cnnCat2.h5"

h5 = h5py.File(filename,'r')


model_weights = h5['model_weights']
optimizer_weights = h5['optimizer_weights']

print(h5.len)


h5.close()