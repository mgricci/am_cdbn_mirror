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
    def __init__(self, RF_size=10,
		       overlap = 2,
		       weight = 1.0,
		       n_gibbs = 32,
		       style='reichert',
		       sigma=1.0, 
		       bias = 0.0,
		       center_filters='yes'):
        # set attributes, including sampling type
	self.RF_size = RF_size
	self.overlap = overlap
	self.in_shape = [2*self.RF_size - self.overlap, self.RF_size]
	self.out_shape = 2
	self.weight = weight
	self.mid = int(np.floor(RF_size / 2.0))
	self.clamp = self.make_stim()
	self.style    = style
	self.n_gibbs  = n_gibbs
	self.sample = self.reichert_sampling if style is 'reichert' else self.windolf_sampling
	self.sigma_sq = sigma**2

	# Filters
	filt = np.zeros((self.RF_size, self.RF_size))
	filt[:, self.mid] = 1.0
	self.filters = [self.weight*filt, self.weight*filt]
	if center_filters == 'yes':
	    self.center()
	elif center_filters == 'other':
	    self.filters[0][self.filters[0]==0] = -1 * self.weight
	self.biases = bias*np.ones((self.out_shape))
    def center(self):
	for f, filt in enumerate(self.filters):
	    self.filters[f] = filt - np.mean(filt)
    def make_stim(self):
	height = 2*self.RF_size - overlap
	width  = self.RF_size
	stim = np.zeros((height, width))
	stim[:, self.mid] = 1.0
	return stim
    
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
	return rates*np.exp(1j*phases)

    def x_in(self, vis):
	if self.style == 'reichert':
            return [(np.sum(np.abs(vis)*filt), np.sum(vis*filt)) for filt in self.filters]
	else:
	    RFs = [vis[i*self.RF_size - i*self.overlap:(i+1)*self.RF_size,:] for i in range(2)]
            return np.array([np.sum(R*filt) for (R, filt) in zip(RFs, self.filters)]) 

    def x_out(self, hid):
	if self.style == 'reichert':
	    return list(np.sum(np.array([(np.abs(h)*filt, h*filt) for (h,filt) in zip(hid, self.filters)]), axis=0))
	else:
	    topdown = [h*filt for (h, filt) in zip(hid, self.filters)]
	    overlap = topdown[0][-1*self.overlap:, :] + topdown[1][:self.overlap, :]
	    return np.concatenate((topdown[0][:-1*self.overlap, :], overlap, topdown[1][self.overlap:, :]))

    def run(self):
	self.v_run = [self.clamp*np.exp(1j*2*np.pi*np.random.rand(self.in_shape[0], self.in_shape[1]))]
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

#TODO Reichert sampling
n_gibbs= 64
RF_size = 40
overlap = 2
centered_net = my_net(RF_size=RF_size,overlap=overlap,weight=20.0, style='windolf', n_gibbs=n_gibbs, sigma=.1, center_filters='yes', bias=0.0)
uncentered_net = my_net(RF_size = RF_size,overlap=overlap,weight=20.0, style='windolf', n_gibbs=n_gibbs, sigma=.1, center_filters='no', bias=0.0)
other_net = my_net(RF_size = RF_size,overlap=overlap,weight=20.0, style='windolf', n_gibbs=n_gibbs, sigma=.1, center_filters='other', bias=0.0)

#clamp[self.mid,:] = np.exp(1j*0)
#clamp[:,self.mid] = np.exp(1j*np.pi)
#clamp = clamp.astype(np.complex64)
#clamp[self.mid,self.mid] = np.exp(1j*np.pi * .75)
all_cphase = []
all_uphase = []
all_ophase = []
for i in range(50):
    cv, ch = centered_net.run()
    uv, uh = uncentered_net.run()
    ov, oh = other_net.run()
    cv = np.array(cv)
    cv = np.expand_dims(cv, -1)
    cv = np.expand_dims(cv, 0)
    cv[0,:,0,-2:,0] = np.array(ch)
    save_cplx_anim('/home/matt/v', cv)
    ipdb.set_trace()

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
