import numpy as np
from numpy.random import vonmises as vm
from scipy.special import expit as sigmoid
from scipy.special import i0 as bessel
from scipy.signal import convolve2d as conv
from test_utils import save_cplx_anim, generate_shape_img, generate_bars_img
import ipdb
# SHAPES BARS EXPERIMENT:

class my_net(object):
    def __init__(self, in_shape=17,
		       weights = [1.0, 1.0, 1.0],
		       n_gibbs = 32,
		       style='reichert',
		       fix_visible_phase=False,
		       sigma = 1.0, 
		       bias = 0.0,
		       center_filters=False):
        # set attributes, including sampling type
	self.in_shape = in_shape
	self.style    = style
	self.n_gibbs  = n_gibbs
	self.sample = self.reichert_sampling if style is 'reichert' else self.windolf_sampling
	self.fix_phase = fix_visible_phase

	# Filters
	filter1      = np.zeros((in_shape, in_shape))
	filts, afilts = self.make_filters()
	self.filters = [w*f for (w,f) in zip(weights, filts)]
	self.adjoint_filters = [w*f for (w,f) in zip(weights, afilts)]
	if center_filters is True: 
	    self.center()
	self.out_shape = self.in_shape - self.filters[0].shape[0] + 1
	self.biases = bias*np.ones((3, self.out_shape, self.out_shape))
	self.sigma_sq = sigma**2
    def center(self):
	for f, filt in enumerate(self.filters):
	    self.filters[f] = filt - np.mean(np.array(filt))
	    self.adjoint_filters[f] = self.adjoint_filters[f] - np.mean(np.array(filt))
    def make_filters(self):
        filter1 = np.array(
    [[1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1]])


        filter2 = np.array(
    [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        filter1 = np.pad(filter1, ((2,3), (2,3)), 'constant')
        filter2 = np.pad(filter2, ((2,3), (0,0)), 'constant')
        filter3  = np.flipud(filter2)
        filters = [filter1, filter2, filter3]
        adjoint_filters = [np.flip(f, 1) for f in filters]
        return filters, adjoint_filters

    def reichert_sampling(self, params, layer):
	if layer == 'hidden':
	    classic = np.array([p[0] for p in params])
	    synchrony = np.array([p[1] for p in params])
	else: 
	    classic = params[0]
	    synchrony = params[1]
	if layer == 'visible':
	    rates = np.abs(self.clamp)
	    phases = np.where(synchrony == 0, np.angle(self.v_run[-1]), np.angle(synchrony))
	else:
	    rates = 1*(np.random.rand(3, self.out_shape, self.out_shape) < sigmoid((.5*classic + .5*np.abs(synchrony) // self.sigma_sq - self.biases))) 
	    phases = np.angle(synchrony)
	return rates*np.exp(1j*phases)

    def windolf_sampling(self, params, layer):
	a = np.abs(params)
	alpha = np.angle(params)

	if layer=='visible':
	    rates = np.abs(self.clamp)
	else:
	    b = bessel(a - self.biases)	/ self.sigma_sq
	    ber_P = b / (1.0 + b)
	    rates = 1*(np.random.rand(3, self.out_shape, self.out_shape) < ber_P)
	
	phases = vm(alpha, a*rates / self.sigma_sq)
	return rates*np.exp(1j*phases)

    def x_in(self, vis):
	if self.style == 'reichert':
            return [(conv(np.abs(vis), filt, mode='valid'), conv(vis, filt, mode='valid')) for filt in self.filters]
	else: 
            return np.array([conv(vis,filt, mode='valid') for filt in self.filters]) 

    def x_out(self, hid):
	hid = np.pad(hid, ((0,0), (self.filters[0].shape[0] - 1, self.filters[0].shape[0] - 1), (self.filters[0].shape[1] - 1, self.filters[0].shape[1] - 1)), mode='constant')
	if self.style == 'reichert':
	    return list(np.sum(np.array([(conv(np.abs(h), filt, mode='valid'), conv(h,filt, mode='valid')) for (h,filt) in zip(hid, self.adjoint_filters)]), axis=0))
	else:
	    return np.sum(np.array([conv(h,filt,mode='valid') for (h,filt) in zip(hid, self.adjoint_filters)]), axis=0)

    def run(self, clamped_visible):
	self.clamp = clamped_visible
	if self.fix_phase:
	    self.v_run = [clamped_visible]
	else:
	    self.v_run = [clamped_visible*np.exp(1j*2*np.pi*np.random.rand(self.in_shape, self.in_shape))]
	self.h_run = [np.zeros((3, self.out_shape, self.out_shape))]
        for i in range(self.n_gibbs):
	    print('Iteration {}'.format(i))
	    hidden_activations = self.x_in(self.v_run[i-1])
	    hidden_samples = self.sample(hidden_activations, 'hidden')
            self.h_run.append(hidden_samples)
	    visible_activations = self.x_out(self.h_run[i+1])
	    visible_samples = self.sample(visible_activations, 'visible')
	    self.v_run.append(visible_samples)

        return self.v_run, self.h_run

net = my_net(in_shape=44, weights=[1.0, 1.0, 1.0], style='reichert', n_gibbs=64, fix_visible_phase=False, sigma=1.0, bias=1.0, center_filters=True)
clamp = generate_shape_img(20,20, pad_width=12)
v, h = net.run(np.squeeze(clamp))
v = np.array(v)
v = np.expand_dims(v, -1)
v = np.expand_dims(v,0)
h = np.array(h)
h = np.transpose(h, axes=(1,0,2,3))
save_cplx_anim('v', v)
save_cplx_anim('h',h, number=3)
print('done')
