import numpy as np
from numpy.random import vonmises as vm
from scipy.special import expit as sigmoid
from scipy.special import i0 as bessel
from scipy.signal import convolve2d as conv
from test_utils import save_cplx_anim, generate_shape_img, generate_bars_img
import ipdb
# SHAPES WITH GLOBAL RF EXPERIMENT:

class my_net(object):
    def __init__(self, in_shape=17,
		       weights = [1.0, 1.0, 1.0],
		       n_gibbs = 32,
		       style='reichert',
		       sigma = 1.0, 
		       bias = 0.0,
		       f_bias = 0.0,
		       center_filters=False):
        # set attributes, including sampling type
	self.in_shape = in_shape
	self.style    = style
	self.n_gibbs  = n_gibbs
	self.weights  = weights
	self.sample = self.reichert_sampling if style is 'reichert' else self.windolf_sampling

	# Filters
	filter1      = np.zeros((in_shape, in_shape))
	filts, afilts = self.make_filters()
	self.filters = [w*f for (w,f) in zip(weights[:-1], filts)]
	self.adjoint_filters = [w*f for (w,f) in zip(weights[:-1], afilts)]
	if center_filters is True: 
	    self.center()
	self.out_shape = self.in_shape - self.filters[0].shape[0] + 1
	self.biases = bias*np.ones((3, self.out_shape, self.out_shape))
	#self.biases = .4*np.array([np.sum(f[f>0])*np.ones((self.out_shape, self.out_shape)) for f in self.filters])
	
	self.f_biases = f_bias*np.ones((3))
	self.sigma_sq = sigma**2
    def center(self):
	for f, filt in enumerate(self.filters):
	    self.filters[f] = filt - np.mean(filt)
	    self.adjoint_filters[f] = self.adjoint_filters[f] - np.mean(filt)
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
	if layer == 'hidden' or layer=='final':
	    classic = np.array([p[0] for p in params])
	    synchrony = np.array([p[1] for p in params])
	else: 
	    classic = params[0]
	    synchrony = params[1]
	if layer == 'visible':
	    rates = np.abs(self.clamp)
	    phases = np.where(classic == 0, np.angle(self.v_run[-1]), np.angle(synchrony))
	elif layer=='hidden':
	    rates = 1*(np.random.rand(3, self.out_shape, self.out_shape) < sigmoid((.5*classic + .5*np.abs(synchrony) / self.sigma_sq - self.biases))) 
	    phases = np.where(classic == 0, np.angle(self.h_run[-1]), np.angle(synchrony))
	elif layer=='final':
	    rates = 1*(np.random.rand(3) < sigmoid((.5*classic + .5*np.abs(synchrony) / self.sigma_sq - self.f_biases))) 
	    phases = np.where(classic == 0, np.angle(self.f_run[-1]), np.angle(synchrony))
	return rates*np.exp(1j*phases)

    def windolf_sampling(self, params, layer):
	a = np.abs(params)
	alpha = np.angle(params)

	if layer=='visible':
	    rates = np.abs(self.clamp)
	elif layer=='hidden':
	    ipdb.set_trace()
	    b = bessel(a - self.biases)	/ self.sigma_sq
	    ber_P = b / (1.0 + b)
	    rates = 1*(np.random.rand(3, self.out_shape, self.out_shape) < ber_P)
	elif layer=='final':
	    b = bessel(a - self.f_biases)/ self.sigma_sq
	    ber_P = b / (1.0 + b)
	    rates = 1*(np.random.rand(3) < ber_P)
	phases = vm(alpha, a*rates / self.sigma_sq)
	return rates*np.exp(1j*phases)

    def x_final_in(self, hid):
        if self.style == 'reichert':
	    return [((self.weights[-1] / self.out_shape**2) * np.sum(np.abs(h)), (self.weights[-1] / self.out_shape**2)*np.sum(h)) for h in hid]
	else: 
	    return np.array([(self.weights[-1] // self.out_shape**2)*np.sum(h) for h in hid])

    def x_final_out(self, final): 
	if self.style == 'reichert':
	    return [((self.weights[-1] / self.out_shape**2)*np.abs(f)*np.ones((self.out_shape, self.out_shape)), (self.weights[-1] / self.out_shape**2)*f*np.ones((self.out_shape, self.out_shape))) for f in final]
	else: 
	    return [f*np.ones((self.out_shape, self.out_shape)) for f in final]

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
	self.v_run = [clamped_visible*np.exp(1j*2*np.pi*np.random.rand(self.in_shape, self.in_shape))]
	self.h_run = [np.zeros((3, self.out_shape, self.out_shape))]
	self.f_run = [np.zeros(3)]
        for i in range(self.n_gibbs):
	    print('Iteration {}'.format(i))
	    hid_up = self.x_in(self.v_run[i-1])
	    hid_down = self.x_final_out(self.f_run[i-1])
	    if self.style == 'reichert':
	        hid_classic_up = np.array(hid_up)[:,0,:,:]
	        hid_synchrony_up = np.array(hid_up)[:,1,:,:]
	        hid_classic_down = np.array(hid_down)[:,0,:,:]
	        hid_synchrony_down = np.array(hid_down)[:,1,:,:]

		total_hid_classic = list(np.abs(hid_classic_up + hid_classic_down))
		total_hid_synchrony = list(hid_synchrony_up + hid_synchrony_down)

		hidden_activations = [(c,s) for (c,s) in zip(total_hid_classic, total_hid_synchrony)]
	    else: 
		hidden_activations = hid_up + hid_down
	    hidden_samples = self.sample(hidden_activations, 'hidden')
            self.h_run.append(hidden_samples)
	    visible_activations = self.x_out(self.h_run[i+1])
	    visible_samples = self.sample(visible_activations, 'visible')
	    self.v_run.append(visible_samples)
	    final_activations = self.x_final_in(self.h_run[i+1])
	    final_samples = self.sample(final_activations, 'final')
	    self.f_run.append(final_samples)

        return self.v_run, self.h_run, self.f_run

#TODO Integrate across channels
net = my_net(in_shape=44, weights=[1.0, 1.0, 1.0, 100.0], style='windolf', n_gibbs=64, sigma=.01, bias=25.0, f_bias=0.0, center_filters=True)
clamp = generate_shape_img(20,20, pad_width=12)
v, h, f = net.run(np.squeeze(clamp))
v = np.array(v)
f = np.array(f)
v = np.expand_dims(v, -1)
v = np.expand_dims(v,0)
h = np.array(h)
h = np.transpose(h, axes=(1,0,2,3))
full = v
full[0,:, 0, -3:, 0] = f
save_cplx_anim('/home/matt/v', full)
save_cplx_anim('/home/matt/h',h, number=3)
print('done')
