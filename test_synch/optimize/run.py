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
free = True
dl = torch.utils.data.DataLoader(datasets.MNIST('../data',
				                train=False,
					        download=False,
						transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((mu,), (sigma,))
					])),
					batch_size = batch_size, shuffle=True)

def evaluate(net, data):
    acc = []
    for (batch, target) in dl:
        out = net.forward(batch.double().cuda())
	y   = torch.argmax(out,dim=1).float().cpu()
 	t   = target.float()
	match = (t==y).float()
        acc.append(torch.mean(match).data.numpy())	
    print('Validation accuracy: {}'.format(np.mean(acc)))

def Energy(z,k=1):
    out = my_net.run_single_cplx(z).squeeze().data.cpu().numpy()
    out_real = out[:10]
    out_imag = out[10:]
    #out_cplx = out_real + 1j*out_imag
    #print(np.abs(out_cplx*sigma + mu).max())
    out_norm = np.sqrt(out_real**2 + out_imag**2)
    Hj = [(1-(j==k)) + (-1)**(1-(j==k))*out_norm[j] for j in range(10)]
    #return -1*np.prod((Hj)) + 1
    h  = -1*Hj[k] + 1
    #target = np.zeros(10)
    #target[k] = 1
    #h = np.sqrt(np.sum((out_norm - target)**2))
    print(h)
    return h
    
def constraint(z, i,free=False):
    ind = np.unravel_index(i, (img_side, img_side))
    c = v_prime[ind] if not free else (1-mu)/sigma
    z_real = z[:img_side**2].reshape(img_side, img_side)
    z_imag = z[img_side**2:].reshape(img_side, img_side)
    z_cplx = z_real + 1j*z_imag
    y = z_cplx[ind]
    return np.abs(c + mu / sigma) - np.abs(y + mu / sigma)

definit(x):
    x = x.astype(np.complex128)
    x = sigma*x + mu
    x*=.25*np.random.rand(x.shape[0], x.shape[1])
    x*=np.exp(2*np.pi*1j*np.random.rand(x.shape[0], x.shape[1]))
    return (x-mu) / sigma

evaluate(my_net, dl)
for (batch, target) in dl:
    #v_prime = batch.data.numpy()[0,0,:,:]
    v_prime = (np.random.rand(28,28) - mu) / sigma
    k = target[0].data.numpy()
    print('Target: {}'.format(k))
    z0 = init(v_prime)
    #z0 = v_prime
    z0_real = np.real(z0)
    z0_imag = np.imag(z0)
    z0_flat = np.concatenate((z0_real.reshape(-1), z0_imag.reshape(-1)), axis=0)

    print('Making constraints...')	
    cons = tuple([{'type' : 'ineq',
		   'fun'  : constraint,
	 	   'args' : (j,free,)} for j in range(img_side**2)])
    print('Optimizing...')
    res = minimize(Energy, z0_flat, method='SLSQP', constraints=cons, options={'disp': True, 'maxiter':max_iter, 'verbose':2, 'ftol':0.0}, args=k)
    cp = res.x
    cp_real = cp[:784].reshape(28,28)
    cp_imag = cp[784:].reshape(28,28)
    cp_cplx = cp_real + 1j*cp_imag
   
    for i, img in enumerate([z0, cp_cplx]):
 	name = 'init' if i == 0 else 'seg'
        fig, ax = plt.subplots()
        cplx_imshow(ax, img, cm=plt.cm.hsv, remap=(mu,sigma))
        plt.savefig('/home/jk/matt/' + name + '.png')
        plt.close()
    ipdb.set_trace() 
    print('hey')
