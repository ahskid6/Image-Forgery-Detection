import time
import numpy as np
import pyfftw

def describe(name,arr):
	try:
		print(name+", shape: "+str(arr.shape)+", dtype: "+str(arr.dtype) \
				+" (min,max): ("+str(np.amin(arr))+", "+str(np.amax(arr))+")" \
				+" (mean,std): ("+str(np.mean(arr))+", "+str(np.std(arr))+")")
	except:
		try:
			print(name+", shape "+str(arr.get_shape())+", type: "+str(type(arr)))
		except:
			print(name+", type: "+str(type(arr)))


def fftscores(arrs, is_already_fft=False):
	if len(arrs.shape) == 2:
		print("fftscores: warning: arrs should be in batch mode, with the 0-axis indexing batch items")
		arrs = arrs.reshape([1,]+list(arrs.shape))
	if is_already_fft:
		absfft = arrs.copy() 
	else:
		truefft = pyfftw.interfaces.numpy_fft.rfft(arrs, axis=2, planner_effort='FFTW_PATIENT', threads=6)
		absfft = np.absolute(truefft)
	fftmax = np.amax(absfft, axis=1, keepdims=True) 
	fftavg = np.mean(absfft, axis=1, keepdims=True) 

	if True:
		print("fftscore is complex")
		score = np.divide(truefft, fftavg+1e-16)
	else:
		print("fftscore is abs-based")
		score = np.divide(absfft, fftavg+1e-16)

	return absfft, score, np.divide(fftmax, fftavg+1e-16), fftavg

