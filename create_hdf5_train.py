import os
from skimage import io
import numpy as np
import h5py
from PIL import Image
folder1 = "train_data/spliced_copymove_NIST/rgb_imgs/"#"/home/adaptix_anik/Documents/MegaProjectWork/Datasets/Bernard/Computer_Vision/spliced_copymove_NIST/rgb_imgs/"
#"train_data/spliced_copymove_NIST/rgb_imgs/"
files1 = os.listdir(folder1)

folder2 = "train_data/spliced_copymove_NIST/masks/"#"/home/adaptix_anik/Documents/MegaProjectWork/Datasets/Bernard/Computer_Vision/spliced_copymove_NIST/masks/"
#"train_data/spliced_copymove_NIST/masks/"
files2 = os.listdir(folder2)

hdf5_path = 'train_data/spliced_copymove_nist_200_next.hdf5'
files1.sort()
files2.sort()

n=len(files1)
print(files1)
print(files2)
train_shape = (n, 256, 256, 3)
test_shape = (n,256,256)

hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset(name="train_img", 
                         shape=train_shape, 
                         compression=None)

hdf5_file.create_dataset(name="train_mask", 
                         shape=test_shape,
                         compression=None)
i=0
for file_name in files1:
    rgb_img = Image.open(folder1+file_name)
    rgb_img = rgb_img.resize((256,256))
    hdf5_file["train_img"][i, ...] = rgb_img
    i=i+1

i=0
for file_name in files2:
    mask_img = Image.open(folder2+file_name)
    mask_img = mask_img.resize((256,256))
    hdf5_file["train_mask"][i, ...] = mask_img
    i=i+1

hdf5_file.close()