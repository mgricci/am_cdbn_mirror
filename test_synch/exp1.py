import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import numpy as np
from numpy.random import vonmises as vm
from scipy.special import expit as sigmoid
from scipy.special import i0 as bessel
from test_utils import save_cplx_anim
import ipdb
# CROSSED BARS EXPERIMENT:

class my_net(object):
    def __init__(self, in_shape=17,
		       weights = [1.0, 1.0],
		       n_gibbs = 32,
		       style='reichert',
		       fix_visible_phase=False, 
		       sigma=1.0, 
		       bias = 0.0,
		       center_filters='yes'):
        # set attributes, including sampling type
	self.in_shape = in_shape
	self.style    = style
	self.n_gibbs  = n_gibbs
	self.sample = self.reichert_sampling if style is 'reichert' else self.windolf_sampling
	self.fix_phase = fix_visible_phase
	self.sigma_sq = sigma**2
	self.mid = int(np.floor(in_shape / 2.0))
	

	# Filters
	filter1      = np.zeros((in_shape, in_shape))
	filter1[self.mid,:] = 1.0
	filter2 = filter1.T
	self.filters = [weights[0]*filter1, weights[1]*filter2]
	if center_filters == 'yes':
	    self.center()
	elif center_filters == 'other':
	    self.filters[0] -= self.filters[1]
	    self.filters[0][self.mid,self.mid] = weights[0]
	    self.filters[1] = np.copy(self.filters[0].T)
	    self.filters[1][self.mid,self.mid] = -weights[1]
	self.out_shape = len(self.filters)
	self.biases = bias*np.ones((self.out_shape))
    def center(self):
	for f, filt in enumerate(self.filters):
	    self.filters[f] = filt - np.mean(filt)
    
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
	    rates = 1*(np.random.rand(self.out_shape) < sigmoid(.5*classic + .5*np.abs(synchrony) - self.biases)) 
	    phases = np.angle(synchrony)

	if self.fix_phase and layer == 'visible':
	    center_phase = phases[self.mid,self.mid]
	    phases = np.angle(self.clamp)
	    phases[self.mid,self.mid] = center_phase

	return rates*np.exp(1j*phases)

    def windolf_sampling(self, params, layer):
	a = np.abs(params)
	alpha = np.angle(params)

	if layer=='visible':
	    rates = np.abs(self.clamp)
	    #phases = np.where(rates>0, vm(alpha, rates*a / self.sigma_sq), np.angle(self.v_run[-1]))
	else:
	    b = bessel(a - self.biases)	/ self.sigma_sq
	    ber_P = b / (1.0 + b)
	    rates = 1*(np.random.rand(self.out_shape) < ber_P)
	    #phases = np.where(rates > 0, vm(alpha, rates*a / self.sigma_sq), np.angle(self.h_run[-1]))
	phases = vm(alpha, rates*a / self.sigma_sq)
	if self.fix_phase and layer == 'visible':
	    center_phase = phases[self.mid,self.mid]
	    phases = np.angle(self.clamp)
	    phases[self.mid,self.mid] = center_phase
	return rates*np.exp(1j*phases)

    def x_in(self, vis):
	if self.style == 'reichert':
            return [(np.sum(np.abs(vis)*filt), np.sum(vis*filt)) for filt in self.filters]
	else: 
            return np.array([np.sum(vis*filt) for filt in self.filters]) 

    def x_out(self, hid):
	if self.style == 'reichert':
	    return list(np.sum(np.array([(np.abs(h)*filt, h*filt) for (h,filt) in zip(hid, self.filters)]), axis=0))
	else:
	    return np.sum(np.array([h*filt for (h,filt) in zip(hid, self.filters)]), axis=0)

    def run(self, clamped_visible):
	self.clamp = clamped_visible
	if self.fix_phase:
	    self.v_run = [clamped_visible]
	else:
	    self.v_run = [clamped_visible*np.exp(1j*2*np.pi*np.random.rand(self.in_shape, self.in_shape))]
	self.h_run = [np.zeros(self.out_shape)]
        for i in range(self.n_gibbs):
	    print('Iteration {}'.format(i))
	    hidden_activations = self.x_in(self.v_run[i-1])
	    hidden_samples = self.sample(hidden_activations, 'hidden')
            self.h_run.append(hidden_samples)
	    visible_activations = self.x_out(self.h_run[i+1])
	    visible_samples = self.sample(visible_activations, 'visible')
	    self.v_run.append(visible_samples)

        return self.v_run, self.h_run

#reichert_net = my_net(weights=[.01 ,.01], n_gibbs=128)
#For Windolf/ Reichert: weights   = [20,20] / [0.01, 0.01]
#	        	n_gibbs   = 128 / 256
#	        	sigma     = 0.1 / 1.0
#	        	bias      = 0.0 / 0.0
#	        	size      = 17  / 17
#	        	n_samples = 50  / 50

n_gibbs=200
size = 19
centered_net = my_net(in_shape=size,weights=[20.0, 20.0], style='windolf', n_gibbs=n_gibbs, fix_visible_phase=False, sigma=.1, center_filters='yes', bias=0.0)
uncentered_net = my_net(in_shape=size,weights=[20.0, 20.0], style='windolf', n_gibbs=n_gibbs, fix_visible_phase=False, sigma=.1, center_filters='no', bias=0.0)
other_net = my_net(in_shape = size,weights=[20.0, 20.0], style='windolf', n_gibbs=n_gibbs, fix_visible_phase=False, sigma=.1, center_filters='other', bias=0.0)
clamp = np.zeros((size, size))
mid = int(np.floor(size / 2.0))
clamp[mid,:] = 1.0
clamp[:,mid] = 1.0

#clamp[self.mid,:] = np.exp(1j*0)
#clamp[:,self.mid] = np.exp(1j*np.pi)
#clamp = clamp.astype(np.complex64)
#clamp[self.mid,self.mid] = np.exp(1j*np.pi * .75)
all_cphase = []
all_uphase = []
all_ophase = []
for i in range(50):
    cv, ch = centered_net.run(clamp)
    uv, uh = uncentered_net.run(clamp)
    ov, oh = other_net.run(clamp)
    #uv = np.array(uv)
    #uv = np.expand_dims(uv, -1)
    #uv = np.expand_dims(uv, 0)
    #uv[0,:,0,-2:,0] = np.array(uh)
    #save_cplx_anim('/home/matt/v', uv)

    cphase_differences = [ np.abs(np.mean(np.angle(vis[mid,:])) - np.mean(np.angle(vis[:,mid]))) for vis in cv]
    uphase_differences = [ np.abs(np.mean(np.angle(vis[mid,:])) - np.mean(np.angle(vis[:,mid]))) for vis in uv]
    ophase_differences = [ np.abs(np.mean(np.angle(vis[mid,:])) - np.mean(np.angle(vis[:,mid]))) for vis in ov]

    all_cphase.append(cphase_differences)
    all_uphase.append(uphase_differences)
    all_ophase.append(ophase_differences)

all_cphase = np.array(all_cphase)
all_uphase = np.array(all_uphase)
all_ophase = np.array(all_ophase)
#PLOT
mean_cphase = np.mean(all_cphase, axis=0)
std_cphase  = np.std(all_cphase, axis=0)
mean_uphase = np.mean(all_uphase, axis=0)
std_uphase  = np.std(all_uphase, axis=0)
mean_ophase = np.mean(all_ophase, axis=0)
std_ophase = np.std(all_ophase, axis=0)
plt.plot(np.array(range(n_gibbs+1)), mean_cphase, color='#CC4F1B')
plt.fill_between(np.array(range(n_gibbs+1)), mean_cphase - std_cphase, mean_cphase + std_cphase, edgecolor = '#CC4F1B', facecolor='#FF9848', alpha=.5)

plt.plot(mean_uphase, color='#1B2ACC')
plt.fill_between(np.array(range(n_gibbs+1)), mean_uphase - std_uphase, mean_uphase + std_uphase, edgecolor = '#1B2ACC', facecolor='#089FFF', alpha=.5)
plt.plot(mean_ophase, color='#3F7F4C')
plt.fill_between(np.array(range(n_gibbs+1)), mean_ophase - std_ophase, mean_ophase + std_ophase, edgecolor='#3F7F4C', color='#7EFF99')
plt.legend(['Centered', 'Uncentered', 'Imbalanced'])
plt.xlabel('Time')
plt.ylabel('Mean phase difference between bars')
plt.savefig('NEWbar_phase_diffs_centered_vs_uncentered.png')


#v = np.array(v)
#v = np.expand_dims(v, -1)
#v = np.expand_dims(v,0)
#h = np.array(h)
#full = v
#full[0,:,0,-2:,0] = h
#save_cplx_anim('v', full)
print('done')
