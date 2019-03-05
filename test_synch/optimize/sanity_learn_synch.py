import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import numpy as np
import torch
import subprocess, os
from torch.autograd import Variable
from tqdm import tqdm
import ipdb
from torchvision import datasets, transforms
from test_utils import cplx_imshow
import subprocess
from models import deep, shallow, shallow_real
from data import bars


def l2_norm(tensor):
    return torch.sqrt(torch.sum(tensor**2))

def weight_normalize(net, across_features=False):
    
    if across_features:
        net.conv1.weight.data.div_(l2_norm(net.conv1.weight.data)**2)
    else:
	for f in range(net.num_features):
            net.conv1.weight.data[f,:,:,:].div_(l2_norm(net.conv1.weight.data[f,:,:,:])**2)

def weight_clip(net):
    w = net.conv1.weight.data
    val  = 1.0 / net.k**2
    cond = torch.abs(w) > 1.0 / net.k**2
    sign = torch.where(w < 0, -1*torch.ones_like(w), torch.ones_like(w))
    fill = val*sign*torch.ones_like(w)
    net.conv1.weight.data = torch.where(cond, fill, w)

def ortho_loss(net):
    w = net.conv1.weight.reshape(net.num_features, -1)
    return torch.sqrt(torch.sum((torch.matmul(w, w.t()) - torch.eye(net.num_features).double().cuda())**2)) / (float(net.num_features))**2

def clip_norm(batch,constraint=None):
    if constraint is None:
        constraint = torch.ones_like(batch).double()[:,0,:,:].unsqueeze(1)
    else:
	constraint = torch.tensor(constraint).cuda()
    batch_norm = torch.sqrt(batch[:,0,:,:]**2 + batch[:,1,:,:]**2).unsqueeze(1) 
    cond = constraint < batch_norm
    batch_norm = torch.where(cond, constraint, batch_norm)
    batch_angle = torch.atan2(batch[:,1,:,:],batch[:,0,:,:]).unsqueeze(1)
    return torch.cat((batch_norm*torch.cos(batch_angle), batch_norm*torch.sin(batch_angle)), dim=1)

def optimize_v(net, batch, lr, constraint=None,num_steps=32, return_progress=False):
    batch_variable = Variable(batch, requires_grad=True)
    #v_optimizer = torch.optim.Adam([batch_variable], lr=lr)
    v_optimizer = torch.optim.SGD([batch_variable], lr=lr, momentum=.9)
    a = []
    for n in range(num_steps):
	v_optimizer.zero_grad()
        activity = net.forward(batch_variable)

	if return_progress:
	    a.append(activity.cpu().data.numpy())
	activity.backward()

	v_optimizer.step()
	batch_variable.data = clip_norm(batch_variable.data, constraint=constraint)

    if return_progress:
        return batch_variable.data, np.array(a)
    else:
	return batch_variable.data

def init(batch):
    init_phase = 2*np.pi * torch.rand_like(batch)
    batch_real = batch*torch.cos(init_phase)
    batch_imag = batch*torch.sin(init_phase)
    
    return torch.cat((batch_real, batch_imag),dim=1)

def run(net,dl, num_epochs=128,num_inner_steps=32,batch_size=64, lrs=[1e-2,1e-2, 1e-2], show_every=10, train=True):
    lr1,lr2, lr3 = lrs
    ma = [] 
    ol = []
    #weight_clip(net)
    weight_normalize(net, across_features=True)
    f_optimizer = torch.optim.Adam(net.parameters(), lr=lr1)
    for t in tqdm(range(num_epochs)):
        for b, (batch, target) in enumerate(dl):
	    batch = batch.double().cuda()
	    f_optimizer.zero_grad()
	    if train:
	        activ = net.forward(batch)
	        activ.backward()
	        if np.isnan(net.conv1.weight.grad.data.cpu().numpy().mean()):
	            ipdb.set_trace()
	        f_optimizer.step()
	        weight_normalize(net,across_features=False)
	    else:
		 optim_activ = net.forward(optim_batch)
	    if t%show_every == 0:
	        disp(net,ma,ol,t, batch=None)
	    ma.append(activ.data.cpu().numpy())
    print('hey')
    return np.array(ma), np.array(ol), optim_batch.data.cpu().numpy()

def disp(net, ma, ol, b, batch=None):
    filters = net.conv1.weight.data.cpu().numpy()
    fig, axes = plt.subplots(4,4)	
    for a, ax in enumerate(axes.reshape(-1)):
	f = filters[a,:,:].squeeze()
        im = ax.imshow(f)
	if a == len(axes.reshape(-1)) - 1:
	    fig.colorbar(im, ax=ax)
    plt.savefig('/home/jk/matt/nat_learn_synch/filters{}.png'.format(b))
    plt.close()
    plt.plot(ma)
    plt.savefig('/home/jk/matt/nat_learn_synch/activities.png')
    plt.close()
    plt.plot(ol)
    plt.savefig('/home/jk/matt/nat_learn_synch/ortho_loss.png')
    plt.close()
    if batch is not None:
        for i in range(5):
            fig, ax = plt.subplots()
	    cplx = batch[i, 0, :,: ] + 1j*batch[i, 1, : ,:]
	    cplx_imshow(ax, cplx, cm=plt.cm.hsv)
	    plt.savefig('/home/jk/matt/nat_learn_synch/synch{}.png'.format(i))
	    plt.close()

def get_ds(data_type):
    if data_type=='bars':
 	return bars(img_side=20,num_bars=1)
    elif data_type=='olshausen':
        return datasets.ImageFolder('/home/jk/matt/oandf',
				   transform=transforms.Compose([transforms.RandomCrop(32),
								transforms.Grayscale(),
								transforms.ToTensor()]))
if __name__ == '__main__':
    #num_batches = 512
    #TODO Adaptive rate
    net_type = 'shallow'
    data_type = 'olshausen'

    batch_size=64
    ds = get_ds(data_type)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    save_path = '/home/jk/matt/net.pt'
    load = False
    train = True
    net = shallow_real
    lrs = [1e-3,1e-1, 0.0]
    num_features=16
    num_epochs = 512
    num_inner_steps = 16
    my_net = net(num_features, k=12).double().cuda()
    if load:
        my_net.load_state_dict(torch.load(save_path))
    clean_figs = True
    if clean_figs:
        subprocess.call('rm /home/jk/matt/nat_learn_synch/* &', shell=True)
    print('Loading data...')

    print('Optimizing...') 
    ma, ol, ob= run(my_net,dl,num_epochs=num_epochs, num_inner_steps = num_inner_steps, lrs=lrs, show_every=10, train=train)
    print('Plotting...')
    disp(my_net, ma, ol, num_epochs, batch=ob)
    torch.save(my_net.state_dict(), save_path)

