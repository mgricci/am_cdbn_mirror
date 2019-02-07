import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import numpy as np
import ipdb
from scipy.optimize import minimize
from scipy.signal import convolve2d
import itertools
from test_utils import cplx_imshow
from tqdm import tqdm

RF_side = 5
kernel = -1*np.ones((RF_side, RF_side))
kernel[:,2]*= -1
kernel/=RF_side
img_side = 9
thresh_plot = True
max_iter = 200
num_reps = 10
hid_side = img_side - RF_side + 1
sub_exp = 1

def forward(z):
    z_real = z[:img_side**2].reshape(img_side, img_side)
    z_imag = z[img_side**2:].reshape(img_side, img_side)
    y = z_real + 1j*z_imag
    return convolve2d(y,kernel, mode='valid')

def pool(z):
    hidden = forward(z)
    moduli = np.abs(hidden)
    pool1_RF  = hidden[:int(np.ceil(hid_side / 2.0)), :]
    pool2_RF  = hidden[int(np.ceil(hid_side / 2.0)):, :]

    pool1_argmax = np.unravel_index(np.argmax(np.abs(pool1_RF)), (hid_side, hid_side))
    pool2_argmax = np.unravel_index(np.argmax(np.abs(pool2_RF)), (hid_side, hid_side))

    pool1 = np.amax(np.abs(pool1_RF))*np.exp(1j*np.angle(pool1_RF[pool1_argmax]))
    pool2 = np.amax(np.abs(pool2_RF))*np.exp(1j*np.angle(pool2_RF[pool2_argmax]))
    return [pool1, pool2]
def Ham(z):
    #print(np.abs(z).max())
    hidden = forward(z)
    return -1 * np.sqrt(np.sum(np.abs(hidden)**2))

def constraint(z, i):
    ind = np.unravel_index(i, (img_side, img_side))
    c = clamp[ind]
    z_real = z[:img_side**2].reshape(img_side, img_side)
    z_imag = z[img_side**2:].reshape(img_side, img_side)
    z_cplx = z_real + 1j*z_imag
    y = z_cplx[ind]
    return c - np.sqrt(np.abs(y))
extrema = []
avg_phase_diff = []
std_phase_diff = []
for i in tqdm(range(img_side)):
    clamp = np.zeros((img_side, img_side))
    clamp[:int(np.ceil(img_side / 2.0)), i] = 1.0
    clamp[int(np.ceil(img_side / 2.0)):, img_side - i - 1] = 1.0
    phase_diff = []
    for n in range(num_reps):
        v0 = clamp*np.exp(1j*2*np.pi*np.random.rand(img_side, img_side))
        v0_real = np.real(v0)
        v0_imag = np.imag(v0)
        z0 = np.concatenate((v0_real.reshape(-1), v0_imag.reshape(-1)))
        cons = tuple([{'type' : 'ineq',
		   'fun'  : constraint,
	 	   'args' : (j,)} for j in range(img_side**2)])
        res = minimize(Ham, z0, method='SLSQP', constraints=cons, options={'disp':True, 'maxiter':max_iter})
        ex = res['x']
        ex_cplx = ex[:img_side**2].reshape(img_side, img_side) + 1j*ex[img_side**2:].reshape(img_side, img_side)
        if sub_exp == 1:
            bar_1_avg_phase = np.mean(np.angle(ex_cplx[:int(np.ceil(img_side / 2.0)),i]))
            bar_2_avg_phase = np.mean(np.angle(ex_cplx[int(np.ceil(img_side / 2.0)):,img_side - i - 1]))
            phase_diff.append(np.abs(bar_1_avg_phase - bar_2_avg_phase))
        elif sub_exp == 2:
	    p = pool(ex)
	    phase_diff.append(np.abs(np.angle(p[0]) - np.angle(p[1])))
    avg_phase_diff.append(np.mean(phase_diff)) 
    std_phase_diff.append(np.std(phase_diff))
np.save('/home/matt/avg_harrison.npy', np.array(avg_phase_diff))
np.save('/home/matt/std_harrison.npy', np.array(std_phase_diff))
t = np.array(range(img_side)) - int(np.floor(img_side / 2.0))
x = np.array(avg_phase_diff)
s = np.array(std_phase_diff)
plt.plot(t,x, color='#CC4F1B')
plt.fill_between(t, x - s, x + s, edgecolor='#CC4F1B', facecolor='#FF9848', alpha=.5)
plt.savefig('/home/matt/geman_style_videos/phase_diff.png')

    
