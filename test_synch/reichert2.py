import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import ipdb
from test_utils import save_cplx_anim, get_shape_cov, generate_shape_img

class reichert_net():
    def __init__(self,
		 num_units,
		 drive_weight = .5,
		 num_steps    = 100,
		 weights      = 'gaussian', 
		 clamp        = True,
		 init_T       = 1.0,
		 min_T        = 1.0,
		 rate_T       = 1.0):

	self.num_units    = num_units
	self.drive_weight = drive_weight
	self.num_steps    = num_steps
	self.clamp        = clamp
	self.init_T       = init_T
	self.current_T    = init_T
	self.min_T        = min_T
	self.rate_T       = rate_T

	if weights == 'gaussian':
	    self.W = np.random.normal(0,1,(self.num_units, self.num_units)) 
	    self.W[np.tril_indices(self.num_units)] = 0
	    self.W += np.triu(self.W).T
	    self.b = np.random.normal(0,1,self.num_units)
	elif weights == 'covariance':
	    print('Generating covariance matrix...')
	    self.W = get_shape_cov(int(np.sqrt(num_units)), int(np.sqrt(num_units)), num_samples=10000)
	    self.b = np.zeros(self.num_units)	
	elif weights == 'ferro':
	    self.W = np.ones((self.num_units, self.num_units))
	    self.b = np.zeros(self.num_units)
	elif weights == 'antiferro':
	    self.W = -1*np.ones((self.num_units, self.num_units))
	    self.b = np.zeros(self.num_units)
        if self.clamp:
	    self.clamped_stimulus = generate_shape_img(int(np.sqrt(num_units)), int(np.sqrt(num_units))).reshape(-1)
    def step(self, state, sample=False):
        post_drive = np.matmul(self.W, state) + np.matmul(self.b, state)
        pre_drive = np.matmul(self.W, np.abs(state)) + np.matmul(self.b, np.abs(state))

	total_drive = self.drive_weight*np.abs(post_drive) + (1-self.drive_weight)*pre_drive
	rate 	    = 1.0 / (1.0 + np.exp(-1*total_drive / self.current_T))

	if not self.clamp:
	    if sample:
	        rate = np.where(rate < np.random.rand(rate.shape[0]), np.zeros(rate.shape[0]), np.ones(rate.shape[0]))
	else:
	    rate = self.clamped_stimulus
	phase = np.angle(post_drive)
	
	return rate*np.exp(1j*phase)

    def run(self, sample=False):
	clamp = self.clamped_stimulus if self.clamp else np.ones(num_units)
        x = clamp*np.exp(1j*2*np.pi*np.random.rand(self.num_units))
	coherence = []
	x_run = [x]
 	for t in range(self.num_steps):
	   x = self.step(x,sample=sample)
	   coherence.append(np.abs(np.mean(x)))
	   x_run.append(x)
	   self.current_T*=self.rate_T
	   self.current_T = min(self.current_T, self.min_T)
        return np.array(coherence), np.array(x_run) 

if __name__=='__main__':
    num_units = 625
    num_steps = 100
    rn = reichert_net(num_units, num_steps=num_steps, drive_weight=.5, weights='covariance', clamp=False)

    ch, xr = rn.run(sample=False)
    xr = np.expand_dims(xr.reshape(num_steps+1, 25,25), 0)

    plt.plot(ch)
    plt.savefig('/home/jk/matt/coherence.png')
    print('Maing videos...')
    save_cplx_anim('/home/jk/matt/xrun', xr)



