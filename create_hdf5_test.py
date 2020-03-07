import os
from skimage import io
import numpy as np
import h5py
from PIL import Image

folder = "test_data/imgs/"
files = os.listdir(folder)
hdf5_path = 'test_data/test_data2.hdf5'

n=len(files)
train_shape = (n, 256, 256, 3)
test_shape = (n,256,256)

hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset(name="test_img", 
                         shape=train_shape, 
                         compression=None)

hdf5_file.create_dataset(name="test_labels", 
                         shape=test_shape,
                         compression=None)
i=0
for file_name in files:
    rgb_img = Image.open(folder+file_name)
    rgb_img = rgb_img.resize((256,256))
    gray_img = rgb_img.convert('L') #L parameter for gray conversion
    
    hdf5_file["test_img"][i, ...] = rgb_img
    hdf5_file["test_labels"][i, ...] = gray_img
    i=i+1
  
hdf5_file.close()