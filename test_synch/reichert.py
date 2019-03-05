import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import ipdb
from test_utils import save_cplx_anim, get_shape_cov

class reichert_net():
    def __init__(self,
		 num_vis, 
		 num_hid,
		 kind, 
		 drive_weight = .5,
		 num_steps    = 100):

        self.num_vis      = num_vis
	self.num_hid      = num_hid
	self.num_units    = num_vis + num_hid
	self.kind         = kind
	self.drive_weight = drive_weight
	self.num_steps    = num_steps

	self.W = np.random.normal(0,1,(self.num_units, self.num_units)) 
	self.W[np.tril_indices(self.num_units)] = 0
	self.W += np.triu(self.W).T
	
	if self.kind == 'restricted':
	    self.W[:self.num_vis, :self.num_hid] = 0
	    self.W[self.num_vis:, self.num_hid:] = 0
	
	self.b = np.random.normal(0,1,self.num_units)


    def step(self, state, sample=False):
        post_drive = np.matmul(self.W, state) + np.matmul(self.b, state)
        pre_drive = np.matmul(self.W, np.abs(state)) + np.matmul(self.b, np.abs(state))

	total_drive = self.drive_weight*np.abs(post_drive) + (1-self.drive_weight)*pre_drive
	rate 	    = 1.0 / (1.0 + np.exp(-1*total_drive))

	if sample:
	    rate = np.where(rate < np.random.rand(rate.shape[0]), np.zeros(rate.shape[0]), np.ones(rate.shape[0]))
	phase = np.angle(post_drive)
	
	return rate*np.exp(1j*phase)

    def run(self, x, sample=False):
        x = x*np.exp(1j*2*np.pi*np.random.rand(self.num_units))
	coherence = []
	x_run = []
 	for t in range(self.num_steps):
	   x = self.step(x,sample=sample)
	   coherence.append(np.abs(np.mean(x)))
	   x_run.append(x
        return np.array(coherence), np.array(x_run) 

if __name__=='__main__':
    kind = 'restricted'
    num_vis = 100
    num_hid = 100
    num_steps = 100
    rn = reichert_net(num_vis, num_hid, kind, num_steps=num_steps, drive_weight=.5)
    x = np.random.rand(num_vis + num_hid)

    ch, xr = rn.run(x,sample=False)
    vr = np.expand_dims(xr[:,:num_vis].reshape(num_steps,8,8), 0)
    hr = np.expand_dims(xr[:,num_vis:].reshape(num_steps,8,8), 0)

    plt.plot(ch)
    plt.savefig('/home/jk/matt/coherence.png')
    print('Maing videos...')
    save_cplx_anim('/home/jk/matt/vrun', vr)
    save_cplx_anim('/home/jk/matt/hrun', hr)



