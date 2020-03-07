import sys
sys.path.append('.')
import os
import h5py
import numpy as np
from scipy import signal
import time
import skimage
import skimage.io, skimage.transform
from skimage.transform import resize
from skimage.util import view_as_windows
import scipy.misc
import scipy.io as sio
from utils import *
from skimage import img_as_uint
import matplotlib.pyplot as plt

thispath = '/home/diksha/Desktop/BtechProject/Image Forgery Detection/'

#hdf5_file = h5py.File('train_data/train_data_feat1.hdf5', mode='w')
hdf5_file = h5py.File('test_data/test_data_feat1.hdf5', mode='w')

#hdf5=h5py.File('train_data/train_data1.hdf5','r')
hdf5=h5py.File('test_data/test_data1.hdf5','r')

#imgs=np.array(hdf5['train_img'])
imgs=np.array(hdf5['test_img'])

hdf5.close()

feat_shape=(np.shape(imgs)[0],64,240)
hdf5_file.create_dataset("feat",feat_shape, np.float32)

for q in range(0,np.shape(imgs)[0]):
	
	im=imgs[q]
	patchsize=32
	rgb_patches = view_as_windows(im,(32,32,3),32)
	rgb_patches = np.squeeze(rgb_patches)
	listofpatches = np.reshape(rgb_patches,(64,32,32,3))
	
	circle_inscribed = False
	numAngles = 10
	theta = np.linspace(0,180,numAngles,endpoint=False)

	def radon_projections_compiled_cuda(patches, thetas, circle_inscribed):
		#sys.path.append(os.path.join(thispath,'build'))
		from pysinogram import BatchRadonTransform
		return np.array(BatchRadonTransform(list(patches), list(thetas), circle_inscribed))

	def radon_projections_skimage_python(patches, thetas, circle_inscribed):
		kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) 
		laplacefilter = lambda xx: np.sqrt(np.fabs(signal.convolve2d(xx, kernel, mode='same', boundary='symm')))
		rgbfilter = lambda xx: np.mean([laplacefilter(xx[:,:,chan]) for chan in range(xx.shape[2])], axis=0)
		myradon = lambda xx: skimage.transform.radon(rgbfilter(xx), theta=theta, circle=circle_inscribed).transpose()
		return np.stack([myradon(patches[ii,...]) for ii in range(patches.shape[0])], axis=0)

	if False:
		t0 = time.time()
		check11 =  radon_projections_compiled_cuda(listofpatches, theta, circle_inscribed)
		t1 = time.time()
		check22 = radon_projections_skimage_python(listofpatches, theta, circle_inscribed)
		t2 = time.time()
		print("Radon projections time, compiled CUDA:  "+str(t1-t0)+" seconds")
		print("Radon projections time, python skimage: "+str(t2-t1)+" seconds")
		describe("check11", check11)
		describe("check22", check22)

		import cv2
		for ii in range(check11.shape[0]):
			checkdiff = np.fabs(check11[ii,:,:] - check22[ii,:,:])
			describe("checkdiff", checkdiff)
			zp = np.zeros((4,check11.shape[2]))
			concat = np.concatenate((check11[ii,:,:], zp, check22[ii,:,:], zp, checkdiff), axis=0)
			
	else:
		radonfunc = radon_projections_skimage_python
		beftime = time.time()
		npresult = radonfunc(listofpatches, theta, circle_inscribed)
		assert len(npresult.shape) == 3, str(npresult.shape)

		absproc = lambda xx: np.expand_dims(np.absolute(xx), axis=-1)
		beftime = time.time()
		_, fftnormed, _, fftavg = fftscores(npresult)

		npresult = absproc(fftnormed) - 1.
		npresult=np.transpose(npresult,(3,0,1,2))		
		npresult=np.reshape(npresult,(64,240))
		print("feature extrating for image # "+ str(q+1)+", with shape-->"+str(np.shape(npresult)))
		describe("npresult", npresult)
		hdf5_file["feat"][q, ...] = npresult[None]

# imgs=np.float32(imgs)
# feat=np.array(hdf5_file["feat"])
# n1=0;n2=16;nb=0
# for i in range(n1,n2):
# 	cmap = plt.get_cmap('bwr')
# 	f,(ax,ax1)=plt.subplots(1,2,sharey=True)
# 	ax.imshow(imgs[i])
# 	ax1.imshow(feat[i])
# 	plt.show()

hdf5_file.close()