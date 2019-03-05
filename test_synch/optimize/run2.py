import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import torch
from torchvision import datasets, transforms
from mnist import Net
from scipy.optimize import minimize
from test_utils import cplx_imshow
import ipdb

PATH   = '/home/jk/matt/mnist_cnn.pt'
img_side = 28
#TODO CUDA
my_net = Net(img_side).double().cuda()
print('Loading model')
my_net.load_state_dict(torch.load(PATH))
my_net.eval()
batch_size = 64
max_iter = 32
mu = 0.1307
sigma = 0.3801
dl = torch.utils.data.DataLoader(datasets.MNIST('../data',
				                train=False,
					        download=False,
						transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((mu,), (sigma,))
					])),
					batch_size = batch_size, shuffle=True)

def Energy(v,k=1):
    v = torch.tensor(v).reshape(1,1,28,28).cuda()
    out = my_net.forward(v).squeeze().data.cpu().numpy()
    target = np.zeros(10)
    target[k] = 1
    h = np.sqrt(np.sum((target-out)**2))
    print(h)
    return h
    
def constraint(v, i):
    v = v.reshape(28,28)
    ind = np.unravel_index(i, (img_side, img_side))
    c = (1 - mu) / sigma 
    y = v[ind]
    return np.abs(c + mu / sigma) - np.abs(y + mu / sigma)

for (batch, target) in dl:
    v_prime = (.5*np.random.rand(28,28) - mu) / sigma
    k = target[0].data.numpy()
    print('Target: {}'.format(k))

    print('Making constraints...')	
    cons = tuple([{'type' : 'ineq',
		   'fun'  : constraint,
	 	   'args' : (j,)} for j in range(img_side**2)])
    print('Optimizing...')
    res = minimize(Energy, v_prime.reshape(-1), method='SLSQP', constraints=cons, options={'disp': True, 'maxiter':max_iter, 'verbose':2}, args=k)
    cp = res.x.reshape(28,28)
   
    for i, img in enumerate([v_prime, cp]):
 	name = 'init' if i == 0 else 'seg'
        fig, ax = plt.subplots()
        plt.imshow(img)
        plt.savefig('/home/jk/matt/' + name + '.png')
        plt.close()
    ipdb.set_trace() 
    print('hey')

