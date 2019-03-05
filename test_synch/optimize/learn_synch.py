import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import torch
import subprocess, os
from torch.autograd import Variable
from tqdm import tqdm
from test_utils import cplx_imshow
import ipdb

class Net(torch.nn.Module):
    def __init__(self,num_features):
	super(Net, self).__init__()
	self.num_features = num_features
	self.conv1 = torch.nn.Conv2d(1,num_features,(5,5), bias=False)
        for f in range(num_features):
	    self.conv1.weight.data[f,:,:,:].div_(torch.sqrt(torch.sum(self.conv1.weight.data[f,:,:,:]**2)))
    def forward(self,x):
	x_real = x[:,0,:,:].unsqueeze(1)
	x_imag = x[:,1,:,:].unsqueeze(1)

        x_real = self.conv1(x_real)
        x_imag = self.conv1(x_imag)
	return torch.sum(x_real**2 + x_imag**2,dim=(1,2,3)).mean()

def img_batch(img_size, batch_size):
   batch = []
   for b in range(batch_size):
       img = np.zeros((1,img_size, img_size)) 
       vertical = np.random.rand() < .5
       loc      = np.random.randint(img_size)
       if vertical:
            img[:,:,loc] = 1.0
       else:
            img[:,loc,:] = 1.0
       batch.append(img)
   return np.array(batch)

def clip_norm(batch,constraint=None):
    if constraint is None:
        constraint = torch.ones_like(batch)
    else:
	constraint = torch.tensor(constraint).cuda()
    batch_norm = torch.sqrt(batch[:,0,:,:]**2 + batch[:,1,:,:]**2) 
    cond = constraint < batch_norm
    batch_norm = torch.where(cond, constraint, batch_norm)
    batch_angle = torch.atan2(batch[:,1,:,:],batch[:,0,:,:])
    return torch.cat((batch_norm*torch.cos(batch_angle), batch_norm*torch.sin(batch_angle)), dim=1)

def optimize_v(net, batch, lr, constraint=None,num_steps=32, return_progress=False):
    batch_variable = Variable(batch, requires_grad=True)
    #v_optimizer = torch.optim.Adam([batch_variable], lr=lr)
    a = []
    for n in range(num_steps):
	net.zero_grad()
	#v_optimizer.zero_grad()
        activity = net.forward(batch_variable)
	if return_progress:
	    a.append(activity.cpu().data.numpy())
	activity.backward()
	ratio = np.abs(batch_variable.grad.data.cpu().numpy()).mean()
	lr2_use = lr / ratio
        batch_variable.data.add_(batch_variable.grad.data*lr2_use)
	#v_optimizer.step()
        batch = batch_variable.cpu().data.numpy()
        batch_cplx = np.expand_dims(batch[:,0,:,:], 1) + np.expand_dims(batch[:,1,:,:], 1)
	ipdb.set_trace()
	batch_variable = Variable(clip_norm(batch_variable, constraint=constraint), requires_grad=True)
	
	#batch_variable = Variable(torch.Tensor(np.concatenate((np.real(batch),np.imag(batch)),axis=1)).cuda().double(), requires_grad=True)
    if return_progress:
        return net.forward(batch_variable), batch_variable.data, np.array(a)
    else:
	return net.forward(batch_variable), batch_variable.data

def init(batch):
    batch = batch.astype(np.complex128)
    #batch*= np.random.rand(batch.shape[0],batch.shape[1],batch.shape[2],batch.shape[3])
    batch*= np.exp(1j*2*np.pi*np.random.rand(batch.shape[0],batch.shape[1],batch.shape[2],batch.shape[3]))
    batch_real = np.real(batch)
    batch_imag = np.imag(batch)
    
    return np.concatenate((batch_real, batch_imag),axis=1)

def run(net,img_size=10, batch_size=64, num_batches=100, lrs=[1e-2,1e-2]):
    lr1,lr2 = lrs
    ma = [] 
    #f_optimizer = torch.optim.Adam([net.conv1.weight], lr=lr1)
    for n in tqdm(range(num_batches)):
	net.zero_grad()
	#f_optimizer.zero_grad()
	batch = img_batch(img_size, batch_size)
	cplx_batch = torch.Tensor(init(batch)).double().cuda()
        max_activations, bv = optimize_v(net, cplx_batch, lr2,constraint=batch,num_steps=64, return_progress=False)
	ratio = np.abs(net.conv1.weight.grad.data.cpu().numpy()).mean()
	lr1_use = lr1 / ratio
	#plt.plot(a)
	#plt.savefig('/home/jk/matt/learn_synch/optim_{}.png'.format(n))
	#plt.close()
	ma.append(max_activations.cpu().data.numpy())
        max_activations.backward()
        #f_optimizer.step()
        net.conv1.weight.data.add_(net.conv1.weight.grad.data*lr1_use)
	for f in range(net.num_features):
	    net.conv1.weight.data[f,:,:,:].div_(torch.sqrt(torch.sum(net.conv1.weight.data[f,:,:,:]**2)))

    return np.array(ma), bv.cpu().numpy()

def disp(net, ma, batch):
    filters = net.conv1.weight.data.cpu().numpy()
    fig, axes = plt.subplots(3,3)	
    for a, ax in enumerate(axes.reshape(-1)):
	f = filters[a,:,:].squeeze()
        im = ax.imshow(f)
	if a == len(axes.reshape(-1)) - 1:
	    fig.colorbar(im, ax=ax)
    plt.savefig('/home/jk/matt/learn_synch/filters.png')
    plt.close()
    plt.plot(ma)
    plt.savefig('/home/jk/matt/learn_synch/activities.png')
    plt.close()
    for i in range(5):
	fig, ax = plt.subplots()
	ipdb.set_trace()
	cplx = batch[i,0,:,:] + 1j*batch[i,1,:,:]
	cplx_imshow(ax, cplx, cm=plt.cm.hsv)
	plt.savefig('/home/jk/matt/learn_synch/synch_{}.png'.format(i))
	plt.close()
	
if __name__ == '__main__':
    num_batches = 1
    #TODO Adaptive rate
    lrs = [1e-2,1e-1]
    num_features=9
    img_size=20
    batch_size=64
    my_net = Net(num_features).double().cuda()

    print('Optimizing...') 
    ma, bv = run(my_net,img_size=img_size,batch_size=batch_size,num_batches=num_batches,lrs=lrs)
    print('Plotting...')
    disp(my_net, ma, bv)

