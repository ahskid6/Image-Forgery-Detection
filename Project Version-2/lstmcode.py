import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import tensorflow.contrib.slim as slim
from matplotlib import pyplot as plt
import os
import math
import h5py
from hilbert import hilbertCurve

nbFilter=32
n_steps = 64 
n_hidden = 64
imSize=256
batch_size = 16

input_layer = tf.placeholder("float", [None, imSize,imSize,3])
freqFeat=tf.placeholder("float", [None, 64,240])

upsample_factor=16
n_classes=2
beta=.01
outSize=16

seq = np.linspace(0,63,64).astype(int)
order3 = hilbertCurve(3)
order3 = np.reshape(order3,(64))
hilbert_ind = np.lexsort((seq,order3))
actual_ind=np.lexsort((seq,hilbert_ind))

weights = {
    'out': tf.Variable(tf.random_normal([64,64,nbFilter]))
}
biases = {
    'out': tf.Variable(tf.random_normal([nbFilter]))
}

def get_kernel_size(factor):
    return 2 * factor - factor % 2

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,filter_size,number_of_classes,number_of_classes), dtype=np.float32)    
    upsample_kernel = upsample_filt(filter_size)    
    for i in range(number_of_classes):        
        weights[:, :, i, i] = upsample_kernel    
    return weights

def segNet(bSize,freqFeat,weights,biases):
    patches=tf.transpose(freqFeat,[1,0,2])
    patches=tf.gather(patches,hilbert_ind)
    patches=tf.transpose(patches,[1,0,2])         
    xCell=tf.unstack(patches, n_steps, 1)
    stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden),output_keep_prob=0.9) for _ in range(2)] )
    out, state = rnn.static_rnn(stacked_lstm_cell, xCell, dtype=tf.float32)
    out=tf.gather(out,actual_ind)
    lstm_out=tf.matmul(out,weights['out'])+biases['out']
    lstm_out=tf.transpose(lstm_out,[1,0,2])
    lstm_out=tf.reshape(lstm_out,[bSize,8,8,nbFilter])
    lstm_out=slim.batch_norm(lstm_out,activation_fn=None)
    lstm_out=tf.nn.relu(lstm_out)
    temp=tf.random_normal([bSize,outSize,outSize,nbFilter])
    uShape1=tf.shape(temp)
    upsample_filter_np = bilinear_upsample_weights(2, nbFilter)
    upsample_filter_tensor = tf.constant(upsample_filter_np)
    lstm_out = tf.nn.conv2d_transpose(lstm_out, upsample_filter_tensor,output_shape=uShape1,strides=[1, 2, 2, 1])
    lstm_out=tf.Print(lstm_out,[lstm_out],"LSTM Output:: ")
    return lstm_out

#lstm_output=segNet(batch_size,freqFeat,weights,biases)

upsampled_logits=segNet(batch_size,freqFeat,weights,biases)

flat_pred=tf.reshape(upsampled_logits,(-1,n_classes))
y_pred=tf.argmax(flat_pred,1)
mask_pred=tf.argmax(upsampled_logits,3)
mask_p=tf.argmax(flat_pred,dimension=1)
lstmout=tf.reshape(mask_p,(-1,256,256))


init = tf.initialize_all_variables()
config=tf.ConfigProto()
config.allow_soft_placement=True
config.log_device_placement=True
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    sess.run(init) 
    feat1=h5py.File('test_data/test_data_feat2.hdf5','r')
    freq1=np.array(feat1["feat"])
    
    hdf5=h5py.File('test_data/test_data2.hdf5','r')
    imgs=np.array(hdf5['test_img'])

    print(type(freq1))
    print("freq "+str(len(freq1)))
    feat1.close()
    hdf5.close()

    step = 1
    iter_tamp=0
    bTamp=16
    epoch_iter_tamp=int(bTamp/16)
    batch_x1=np.zeros((batch_size,64,240))
    step=(bTamp/batch_size)
    # print('Step'+step)
    while  step > 0:        
        if (iter_tamp % epoch_iter_tamp)==0:
            iter_tamp=0
            arr_ind=np.arange(bTamp)
            np.random.shuffle(arr_ind)
            arr_ind=arr_ind
            fr2=freq1[arr_ind, ...]
            fr2=fr2[:(int)(np.shape(arr_ind)[0]/bTamp)*bTamp,...]       
        batch_x1[0:16,...]= fr2[(iter_tamp*bTamp):min((iter_tamp+1)*bTamp,bTamp),...]
        iter_tamp+=1
        res=sess.run(mask_pred, feed_dict={freqFeat: batch_x1})
        step=step-1
        n1=0;n2=batch_size;nb=0
        for i in range(n1,n2):
            cmap = plt.get_cmap('bwr')
            f,(ax,ax1)=plt.subplots(1,2,sharey=True)
            # ax.imshow(freq1[i])
            # ax1.imshow(res[i])
            # #orig.imshow(imgs[i])
            # plt.show()
            print(tf.shape(res,out_type=tf.dtypes.int32,name=None))
            print(res)
            # print("step"+str(step))
        

