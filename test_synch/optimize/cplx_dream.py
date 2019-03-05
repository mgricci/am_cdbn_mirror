import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import torch
from torchvision import datasets, transforms
from mnist import Net
import scipy.ndimage as nd
from torch.autograd import Variable
from test_utils import cplx_imshow
from tqdm import tqdm
import ipdb
import subprocess
import os

CKPT_PATH='/home/jk/matt/mnist_cnn.pt'
SAVE_PATH='/home/jk/matt/cplx_dreams'
clean_save_path = True
if clean_save_path is True:
    clean_string = 'rm {}/*.png &'.format(SAVE_PATH)
    subprocess.call(clean_string,shell=True)
img_side = 28
my_net = Net(img_side).double().cuda()
print('Loading model')
my_net.load_state_dict(torch.load(CKPT_PATH))
batch_size = 64
max_iter = 1000
save_every = 10
mu = 0.1307
sigma = 0.3801
origin = -1*mu/sigma
lr = 1e-2

dl = torch.utils.data.DataLoader(datasets.MNIST('../data', 
						train=False,
						download=False,
						transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((mu,), (sigma,))
					])),
				batch_size=batch_size, shuffle=True)

def Energy1(out,target=1):
    out_norm = torch.sqrt(out[:10]**2 + out[10:]**2)
    return -out_norm[target] + 1

def Energy2(out,target=1):
    out_norm = torch.sqrt(out[:10]**2 + out[10:]**2)
    target_vector = np.zeros(10)
    target_vector[target] = 1
    target_vector = torch.tensor(target_vector).cuda()
    return torch.sqrt(torch.sum((out_norm - target_vector)**2))

def clip_norm(z,constraint=None):
    if constraint is None:
	constraint = np.ones_like(z)
    else:
	constraint = constraint*sigma + mu
    cond = constraint < np.abs(z*sigma + mu)

    z_norm = np.where(cond, constraint, np.abs(z*sigma + mu))
    z_angle = np.angle(z*sigma + mu)
    constrained_z = z_norm*np.exp(1j*z_angle)
    return (constrained_z - mu)/sigma

def init(v,sector=2*np.pi):
    v = v.astype(np.complex128)
    v = v*sigma + mu
    v*= .1*np.random.rand(v.shape[0], v.shape[1], v.shape[2], v.shape[3])
    init_phase = sector*np.random.rand(v.shape[0],v.shape[1],v.shape[2],v.shape[3]) - sector / 2
    v*= np.exp(1j*init_phase)
    return (v-mu) / sigma
    
def run(z0,k, model, energy=Energy1, constraint=None):
    z0_real = np.real(z0).reshape(1, 1, img_side, img_side)
    z0_imag = np.imag(z0).reshape(1, 1, img_side, img_side) 
    z0_cplx = torch.tensor(np.concatenate((z0_real, z0_imag), axis=1)).cuda()
	
    energies = []
    for i in tqdm(range(max_iter)):

    	z0_variable = Variable(z0_cplx, requires_grad=True)
        model.zero_grad()

        out = model.forward_cplx(z0_variable).squeeze(0)
        E = energy(out,target=k)
	energies.append(E.cpu().data.numpy())
        E.backward()
        ratio = np.abs(z0_variable.grad.data.cpu().numpy()).mean()
        lr_use = lr / ratio
        z0_variable.data.sub_(z0_variable.grad.data * lr_use)
        z0_cplx = z0_variable.data.cpu().numpy()  # b, c, h, w
	
	z0 = np.expand_dims(z0_cplx[:,0,:,:] + 1j*z0_cplx[:,1,:,:], axis=0)
	z0 = clip_norm(z0,constraint=constraint)
	
	# Shape for input
	z0_real = np.real(z0)
	z0_imag = np.imag(z0)
	z0_cplx = torch.tensor(np.concatenate((z0_real, z0_imag), axis=1)).cuda()

        if i == 0 or (i + 1) % save_every == 0:
	    fig, ax = plt.subplots()
            cplx_imshow(ax,z0,remap=(mu,sigma))
	    plt.savefig(os.path.join(SAVE_PATH, 'dream%04d.png' % i))
	    plt.close()
    return z0, np.array(energies)

def make_gif():
    process_string = 'convert -delay 10 -loop 0 {}/*.png {}/animation.gif &'.format(SAVE_PATH,SAVE_PATH)
    subprocess.call(process_string,shell=True)

if __name__=='__main__':
    for (batch, target) in dl:
	batch_array = batch.cpu().data.numpy()
	k = target[0].cpu().data.numpy()
        v_prime = np.expand_dims(batch[0,:,:,:], axis=0)
	#v_prime = (np.random.rand(1,1,28,28) - mu) / sigma
        z0 = init(v_prime,sector=2*np.pi)
	print('Optimizing')
        cp, energies = run(z0, k, my_net, constraint=v_prime,energy=Energy1) 
	plt.plot(energies)
	plt.ylim([0,1])
	plt.savefig(os.path.join(SAVE_PATH,'energy.png'))
	plt.close()
	make_gif()
	ipdb.set_trace()
