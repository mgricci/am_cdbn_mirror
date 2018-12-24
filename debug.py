import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib import animation as anim
import numpy as np
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.examples.tutorials.mnist import input_data
import cdbn_backup as cdbn
import cplx_cdbn
from cplx_crbm import *
import os
import ipdb
from colorsys import hls_to_rgb

""" --------------------------------------------
    ------------------- DATA -------------------
    -------------------------------------------- """
def colorize(z):
    if len(z.shape) > 2:
	z = np.squeeze(z)
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi) / (2* np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h,l,s)
    c = np.stack(c, axis=-1)
    return c

class MNIST_HANDLER(object):
  
  def __init__(self, data):
    self.num_training_example = data.train.num_examples
    self.num_test_example     = data.test.num_examples
    self.training_data , self.training_labels = data.train.next_batch(self.num_training_example)
    self.test_data , self.test_labels        = data.test.next_batch(self.num_test_example)
    self.whiten               = False
    self.training_index       = -20
    self.test_index           = -20
    
    
  def do_whiten(self):
    self.whiten         = True
    data_to_be_whitened = np.copy(self.training_data)
    mean                = np.sum(data_to_be_whitened, axis = 0)/self.num_training_example
    mean                = np.tile(mean,self.num_training_example)
    mean                = np.reshape(mean,(self.num_training_example,784))
    centered_data       = data_to_be_whitened - mean                
    covariance          = np.dot(centered_data.T,centered_data)/self.num_training_example
    U,S,V               = np.linalg.svd(covariance)
    epsilon = 1e-5
    lambda_square       = np.diag(1./np.sqrt(S+epsilon))
    self.whitening_mat  = np.dot(np.dot(U, lambda_square), V)    
    self.whitened_training_data  = np.dot(centered_data,self.whitening_mat)
    
    data_to_be_whitened = np.copy(self.test_data)
    mean                = np.sum(data_to_be_whitened, axis = 0)/self.num_test_example
    mean                = np.tile(mean,self.num_test_example)
    mean                = np.reshape(mean,(self.num_test_example,784))
    centered_data       = data_to_be_whitened - mean  
    self.whitened_test_data  = np.dot(centered_data,self.whitening_mat)

    
  def next_batch(self, batch_size, type = 'train'):
    if type == 'train':
      if self.whiten:
        operand = self.whitened_training_data
      else:
        operand = self.training_data
      operand_bis = self.training_labels
      self.training_index = (batch_size + self.training_index) % self.num_training_example
      index = self.training_index
      number = self.num_training_example
    elif type == 'test':
      if self.whiten:
        operand = self.whitened_test_data
      else:
        operand = self.test_data
      operand_bis = self.test_labels
      self.test_index = (batch_size + self.test_index) % self.num_test_example
      index = self.test_index
      number = self.num_test_example

    if index + batch_size > number:
      part1 = operand[index:,:]
      part2 = operand[:(index + batch_size)% number,:]
      result = np.concatenate([part1, part2])
      part1 = operand_bis[index:,:]
      part2 = operand_bis[:(index + batch_size)% number,:]
      result_bis = np.concatenate([part1, part2])
    else:
      result = operand[index:index + batch_size,:]
      result_bis = operand_bis[index:index + batch_size,:]
    return result, result_bis
        

mnist_dataset = MNIST_HANDLER(input_data.read_data_sets('data', one_hot=True))
#mnist_dataset.do_whiten()
sess = tf.Session()




""" ---------------------------------------------
    ------------------- MODEL -------------------
    --------------------------------------------- """

my_cdbn = cdbn.CDBN('mnist_cdbn', 20, os.path.expanduser('~/dbn_figs/log'), mnist_dataset, sess, verbosity = 2)

my_cdbn.add_layer('layer_1', fully_connected = False, v_height = 28, v_width = 28, v_channels = 1, f_height = 11, f_width = 11, f_number = 40, 
               init_biases_H = -3, init_biases_V = 0.01, init_weight_stddev = 0.01, 
               gaussian_unit = False, gaussian_variance = 0.2, 
               prob_maxpooling = True, padding = True, 
               learning_rate = 0.00005, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
               weight_decay = 2.0, sparsity_target = 0.003, sparsity_coef = 0.1)

my_cdbn.add_layer('layer_2', fully_connected = False, v_height = 14, v_width = 14, v_channels = 40, f_height = 7, f_width = 7, f_number = 40, 
               init_biases_H = -3, init_biases_V = 0.025, init_weight_stddev = 0.025, 
               gaussian_unit = False, gaussian_variance = 0.2, 
               prob_maxpooling = True, padding = True, 
               learning_rate = 0.0025, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
               weight_decay = 0.1, sparsity_target = 0.1, sparsity_coef = 0.1)

my_cdbn.add_layer('layer_3', fully_connected = True, v_height = 1, v_width = 1, v_channels = 40*7*7, f_height = 1, f_width = 1, f_number = 200, 
               init_biases_H = -3, init_biases_V = 0.025, init_weight_stddev = 0.025, 
               gaussian_unit = False, gaussian_variance = 0.2, 
               prob_maxpooling = False, padding = False, 
               learning_rate = 0.0025, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
               weight_decay = 0.1, sparsity_target = 0.1, sparsity_coef = 0.1)

my_cdbn.add_softmax_layer(10, 0.1)

my_cdbn.lock_cdbn()


""" ---------------------------------------------
    ------------------ TRAINING -----------------
    --------------------------------------------- """
my_cdbn.manage_layers([], ['layer_1','layer_2','layer_3'], [10000,10000,10000], [1,1,1], 20000, restore_softmax = True, fine_tune = True)

# Init complex net
ccdbn = cplx_cdbn.ComplexCDBN.init_from_cdbn(my_cdbn)

# my_cdbn.do_eval()
start_batch = mnist_dataset.next_batch(20)[0] + 0j

#### LET'S DEBUG ####
which_debug_layer = 0

layer = ccdbn.layers[which_debug_layer]
noise_batch = np.random.binomial(1, 0.5, layer.input) + 0j
input_placeholder = tf.placeholder(tf.complex64, layer.input)
topdown_signal = None
result = 'both'
#conv = complex_conv2d(input_placeholder, layer.kernels, [1,1,1,1], padding='SAME')
#a = tf.abs(conv)
#bessels = bessel_i0(a)
#custom_kernel = tf.constant(1.0, shape=[2,2,layer.f_number, 1])
#sum_bessels = tf.nn.depthwise_conv2d(bessels, custom_kernel, [1,2,2,1], padding='VALID')
#bessel_sftmx_denom = tf.add(1.0, sum_bessels)
#ret_kernel = np.zeros((2,2,layer.f_number, layer.f_number))
#for i in range(2):
#    for j in range(2):
#        for k in range(layer.f_number):
#	    ret_kernel[i,j,k,k] = 1
#custom_kernel_bis = tf.constant(ret_kernel, dtype=tf.float32)
#upsampled_denom = tf.nn.conv2d_transpose(bessel_sftmx_denom, custom_kernel_bis, 
#	(layer.batch_size, layer.hidden_height, layer.hidden_width, layer.f_number), 
#	strides= [1,2,2,1], padding='VALID', name=None)
#hid_cat_P = tf.div(bessels, upsampled_denom)
#pool_P = tf.subtract(1.0, tf.div(1.0, bessel_sftmx_denom))

#xin, ab, bes, sb, bd, ud = sess.run([conv, a, bessels, sum_bessels, bessel_sftmx_denom, upsampled_denom], feed_dict = {input_placeholder: 1e5*start_batch.reshape(20,28,28,1)})
hid_probs, pool_probs = layer.dbn_infer_probability(input_placeholder, topdown_signal=topdown_signal, result=result)

hid_rates, pool_rates = tf.py_func(layer._dbn_maxpool_sample_helper, [hid_probs, pool_probs], (tf.float32, tf.float32))
hid_rates.set_shape([layer.batch_size, layer.hidden_height, layer.hidden_width, layer.f_number])
pool_rates.set_shape([layer.batch_size, layer.hidden_height // 2, layer.hidden_width // 2, layer.f_number])
hid_vm_mu, hid_vm_kappa = layer.dbn_infer_vm_params(input_placeholder, hid_rates, topdown_signal=topdown_signal)
hp, pp, hr, pr, hm, hk = sess.run([hid_probs, pool_probs, hid_rates, pool_rates, hid_vm_mu, hid_vm_kappa], feed_dict = {input_placeholder: 1e10*start_batch.reshape(20,28,28,1)})
ipdb.set_trace()
print('Done!')
