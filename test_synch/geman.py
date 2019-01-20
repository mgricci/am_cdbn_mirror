import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
from numpy.random import vonmises as vm
from scipy.special import expit as sigmoid
from scipy.special import i0 as bessel
from scipy.signal import convolve2d as conv
import skimage.util as imx
from test_utils import save_cplx_anim, generate_shape_img, generate_bars_img
import ipdb
# Colinearity detector, so far without a third pooling layer

class my_net(object):
    def __init__(self, in_shape=17,
		       weights = [1.0],
		       kernel_size = 5,
		       conv_stride = 5,
		       n_gibbs = 32,
		       sigma = 1.0, 
		       bias = 0.0):
        # set attributes, including sampling type
	self.in_shape    = in_shape
	self.n_gibbs     = n_gibbs
	self.conv_stride = conv_stride
	# Filters
	self.make_filters(kernel_size, weights[0])
	self.out_shape = int(np.ceil((self.in_shape - self.filters[0].shape[0] + 1) / float(self.conv_stride)))
	self.biases = bias*np.ones((1, self.out_shape, self.out_shape))
	self.sigma_sq = sigma**2
    def make_filters(self, kernel_size=5, weight=1.0):
        filter1 = -5*np.ones((kernel_size, kernel_size))
	#filter1 = np.zeros((kernel_size, kernel_size))
	mid = int(np.floor(kernel_size / 2.))
	filter1[:,mid] = weight
	filter1 -= np.mean(filter1)
	self.filters = [filter1]
        self.adjoint_filters = [np.flip(f, (0,1)) for f in self.filters]

    def windolf_sampling(self, params, layer):
	a = np.abs(params)
	alpha = np.angle(params)

	if layer=='visible':
	    rates = np.abs(self.clamp)
	else:
	    b = bessel(a - self.biases)	/ self.sigma_sq
	    ber_P = b / (1.0 + b)
	    rates = 1*(np.random.rand(1, self.out_shape, self.out_shape) < ber_P)
	
	phases = vm(alpha, a*rates / self.sigma_sq)
	return rates*np.exp(1j*phases)

    def x_in(self, vis):
        return np.array([conv(vis,filt, mode='valid') for filt in self.filters])

    def x_out(self, hid):
	hid = np.pad(hid, ((0,0), (self.filters[0].shape[0] - 1, self.filters[0].shape[0] - 1), (self.filters[0].shape[1] - 1, self.filters[0].shape[1] - 1)), mode='constant')
	return np.sum(np.array([conv(h,filt,mode='valid') for (h,filt) in zip(hid, self.adjoint_filters)]), axis=0)

    def run(self, clamped_visible):
	self.clamp = clamped_visible
	self.v_run = [clamped_visible*np.exp(1j*2*np.pi*np.random.rand(self.in_shape, self.in_shape))]
	self.h_run = [np.zeros((1, self.out_shape, self.out_shape))]
        for i in range(self.n_gibbs):
	    #print('Iteration {}'.format(i))
	    hidden_activations = self.x_in(self.v_run[i-1])
	    hidden_samples = self.windolf_sampling(hidden_activations, 'hidden')
            self.h_run.append(hidden_samples)
	    visible_activations = self.x_out(self.h_run[i+1])
	    visible_samples = self.windolf_sampling(visible_activations, 'visible')
	    self.v_run.append(visible_samples)

        return self.v_run, self.h_run

kernel_size = 4
pad_size  = 0
im_size = 11
n_gibbs = 32
net = my_net(in_shape=im_size + 2*pad_size, kernel_size=kernel_size,weights=[20.0],conv_stride=1,  n_gibbs=n_gibbs, sigma=1, bias=0.0)
phase_diffs = []
num_trials = 25
for t in range(num_trials):
    per_trial_phase_diffs = []
    print('Trial {}'.format(t))
    for i in range(im_size):
        stim = np.zeros((im_size, im_size))
        stim[:int(im_size / 2.0), i] = 1.0
        stim[int(im_size / 2.0):, -1*(i+1)] = 1.0
	stim = np.pad(stim, ((pad_size, pad_size),(pad_size, pad_size)), 'constant')
        #stim[:,4] = 1.0
        v, h= net.run(stim)
        #v, h = np.array(v), np.array(h)
        #h = h[:,0,:,:]
        #v  = np.expand_dims(np.expand_dims(v,0),-1)
        #h = np.expand_dims(np.expand_dims(h,0),-1)
        #save_cplx_anim('/home/matt/geman_style_videos/v{}'.format(i), v)
        #save_cplx_anim('/home/matt/geman_style_videos/h{}'.format(i),h, number=1)
        final_phase_1 = np.mean(np.angle(v[-1][:int(im_size / 2.0), i]))
        final_phase_2 = np.mean(np.angle(v[-1][int(im_size / 2.0):, -1*(i + 1)]))
        per_trial_phase_diffs.append(np.abs(final_phase_1 - final_phase_2))
    phase_diffs.append(per_trial_phase_diffs)
x = np.array([im_size - 1 - 2*j for j in range(im_size)])
phase_diffs = np.array(phase_diffs)
mean_phase_diffs = np.mean(phase_diffs, axis=0)
std_phase_diffs  = np.std(phase_diffs, axis=0)
plt.plot(x, np.array(mean_phase_diffs), color='#CC4F1B')
plt.fill_between(x, mean_phase_diffs - std_phase_diffs, mean_phase_diffs + std_phase_diffs, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.ylim([0,2*np.pi])
plt.title('Bar phase difference at convergence')
plt.savefig('/home/matt/geman_style_videos/phase_diffs.png')
print('done')
