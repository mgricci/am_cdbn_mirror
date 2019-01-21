import tensorflow as tf
import numpy as np
import cdbn_backup as cdbn
import cplx_cdbn
import os
import ipdb
from utils import *
from data import *

shape_dataset = SHAPE_HANDLER()
use_cpu = True
#mnist_dataset = MNIST_HANDLER(input_data.read_data_sets('data', one_hot=True))
#mnist_dataset.do_whiten()
if use_cpu:
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)
else:
    sess = tf.Session()


""" ---------------------------------------------
    ------------------- MODEL -------------------
    --------------------------------------------- """

my_cdbn = cdbn.CDBN('shapes_cdbn', 64, os.path.expanduser('~/dbn_figs/log'), shape_dataset, sess, verbosity = 2, display=True)

my_cdbn.add_layer('layer_1', fully_connected = False, v_height = 20, v_width = 20, v_channels = 1, f_height =7, f_width = 7, f_number = 3, 
               init_biases_H = -3, init_biases_V = 0.01, init_weight_stddev = 0.01, 
               gaussian_unit = False, gaussian_variance = 0.2, 
               prob_maxpooling = True, padding = False, 
               learning_rate = 0.00005, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
               weight_decay = 3.0, sparsity_target = 0.003, sparsity_coef = 0.1)
my_cdbn.add_layer('layer_2', fully_connected = False, v_height = 14, v_width = 14, v_channels = 3, f_height = 8, f_width = 8, f_number = 10, 
               init_biases_H = -3, init_biases_V = 0.025, init_weight_stddev = 0.025, 
               gaussian_unit = False, gaussian_variance = 0.2, 
               prob_maxpooling = True, padding = False, 
               learning_rate = 0.0025, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
               weight_decay = 0.1, sparsity_target = 0.1, sparsity_coef = 0.1)
my_cdbn.add_layer('layer_3', fully_connected = True, v_height = 1, v_width = 1, v_channels = 10*7*7, f_height = 1, f_width = 1, f_number = 676, 
               init_biases_H = -3, init_biases_V = 0.025, init_weight_stddev = 0.025, 
               gaussian_unit = False, gaussian_variance = 0.2, 
               prob_maxpooling = False, padding = False, 
               learning_rate = 0.0025, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
               weight_decay = 0.1, sparsity_target = 0.1, sparsity_coef = 0.1)
#my_cdbn = cdbn.CDBN('mnist_cdbn',20, os.path.expanduser('~/dbn_figs/log'), mnist_dataset, sess, verbosity = 2)
#my_cdbn.add_layer('layer_1', fully_connected = False, v_height = 28, v_width = 28, v_channels = 1, f_height = 11, f_width = 11, f_number = 2, 
#               init_biases_H = -3, init_biases_V = 0.01, init_weight_stddev = 0.01, 
#               gaussian_unit = False, gaussian_variance = 0.2, 
#               prob_maxpooling = True, padding = True, 
#               learning_rate = 0.00005, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
#               weight_decay = 2.0, sparsity_target = 0.003, sparsity_coef = 0.1)

#my_cdbn.add_layer('layer_2', fully_connected = False, v_height = 14, v_width = 14, v_channels = 2, f_height = 7, f_width = 7, f_number = 4, 
#               init_biases_H = -3, init_biases_V = 0.025, init_weight_stddev = 0.025, 
#              gaussian_unit = False, gaussian_variance = 0.2, 
#              prob_maxpooling = True, padding = True, 
#               learning_rate = 0.0025, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
#               weight_decay = 0.1, sparsity_target = 0.1, sparsity_coef = 0.1)

#my_cdbn.add_layer('layer_3', fully_connected = True, v_height = 1, v_width = 1, v_channels = 4*7*7, f_height = 1, f_width = 1, f_number = 200, 
#               init_biases_H = -3, init_biases_V = 0.025, init_weight_stddev = 0.025, 
#               gaussian_unit = False, gaussian_variance = 0.2, 
#               prob_maxpooling = False, padding = False, 
#               learning_rate = 0.0025, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
#               weight_decay = 0.1, sparsity_target = 0.1, sparsity_coef = 0.1)

#my_cdbn.add_softmax_layer(10, 0.1)

my_cdbn.lock_cdbn()

""" ---------------------------------------------
    ------------------ TRAINING -----------------
    --------------------------------------------- """
my_cdbn.manage_layers([],['layer_1','layer_2','layer_3'],[2000,2000,3000], [1,1,1], 20000, restore_softmax = False, fine_tune = True)
my_cdbn.save_feature_figs()
#my_cdbn.do_eval()

# Init complex net
init_sigma=.9
min_sigma=.1
sigma_rate=.998
ccdbn = cplx_cdbn.ComplexCDBN.init_from_cdbn(my_cdbn, init_sigma=init_sigma, min_sigma=min_sigma, sigma_rate=sigma_rate)

#my_cdbn.do_eval()
#start_batch = mnist_dataset.next_batch(20)[0] + 0j
#start_batch = np.ones((20,28,28,1))
#rand_phases= np.exp(1j* 2*np.pi*np.random.rand(start_batch.shape[0], start_batch.shape[1]))
#start_batch*=rand_phases
#combo_batch = mnist_dataset.next_batch(20)[0] + mnist_dataset.next_batch(20)[0]
#combo_batch[combo_batch > 1] = 1
#cplx_combo_batch=combo_batch + 0j
#cplx_combo_batch*=rand_phases
#noise_batch = np.random.binomial(1, 0.5, start_batch.shape) + 0j
#noise_batch*= rand_phases
shape_batch = shape_dataset.next_batch(64)[0] + 0j
rand_phase = np.exp(1j*2*np.pi*np.random.rand(shape_batch.shape[0], shape_batch.shape[1], shape_batch.shape[2], shape_batch.shape[3]))
shape_batch = shape_batch*rand_phase
v, hs, ps = ccdbn.dbn_gibbs(shape_batch, 32, clamp=True)
ipdb.set_trace()
v = v.squeeze().swapaxes(0, 1)
ps = [r.squeeze().swapaxes(0, 1)[:, ..., 0] for r in ps]
hs = [r.squeeze().swapaxes(0, 1)[:, ..., 0] for r in hs]

print('Making animations...')
anim_type = 'mp4'
save_cplx_anim('/home/matt/dbn_figs/new_model/v_', v, number = 10, type=anim_type)
for i, h in enumerate(hs[:-1]):
    save_cplx_anim('/home/matt/dbn_figs/new_model/h%d_' % i, h, number = 5, type=anim_type)

for i, p in enumerate(ps[:-1]):
  save_cplx_anim('/home/matt/dbn_figs/new_model/p%d_' % i, p, number = 5, type=anim_type)
