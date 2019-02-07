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
v_side = 9
thresh_plot = True
num_reps = 10
h_side = v_side - RF_side + 1
sub_exp = 1

def forward(z):
    v = z[:2*v_side**2]
    v_real = v[:v_side**2].reshape(v_side, v_side)
    v_imag = v[v_side**2:].reshape(v_side, v_side)
    y = v_real + 1j*v_imag
    return convolve2d(y,kernel, mode='valid')

def pool(z):
    hidden = forward(z)
    moduli = np.abs(hidden)
    pool1_RF  = hidden[:int(np.ceil(h_side / 2.0)), :]
    pool2_RF  = hidden[int(np.ceil(h_side / 2.0)):, :]

    pool1_argmax = np.unravel_index(np.argmax(np.abs(pool1_RF)), (h_side, h_side))
    pool2_argmax = np.unravel_index(np.argmax(np.abs(pool2_RF)), (h_side, h_side))

    pool1 = np.amax(np.abs(pool1_RF))*np.exp(1j*np.angle(pool1_RF[pool1_argmax]))
    pool2 = np.amax(np.abs(pool2_RF))*np.exp(1j*np.angle(pool2_RF[pool2_argmax]))
    return [pool1, pool2]

def Ham(z):
    drive = forward(z)
    hidden_flat = z[2*v_side**2:]
    hidden_real = hidden_flat[:h_side**2]
    hidden_imag = hidden_flat[h_side**2:]
    hidden      = hidden_real.reshape((h_side, h_side)) + 1j*hidden_imag.reshape((h_side, h_side))
    return np.sqrt(np.sum(np.abs(drive*hidden)**2))

def constraint(z, i):
    v = z[:2*v_side**2]
    ind = np.unravel_index(i, (v_side, v_side))
    c = clamp[ind]
    v_real = v[:v_side**2].reshape(v_side, v_side)
    v_imag = v[v_side**2:].reshape(v_side, v_side)
    v_cplx = v_real + 1j*v_imag
    y = v_cplx[ind]
    return c - np.sqrt(np.abs(y))

def init(v_side,h_side,i):
    v0 = clamp*np.exp(1j*2*np.pi*np.random.rand(v_side, v_side))
    v0_real = np.real(v0)
    v0_imag = np.imag(v0)

    h0 = np.exp(1j*2*np.pi*np.random.rand(h_side, h_side))
    h0_real = np.real(h0)
    h0_imag = np.imag(h0)

    return np.concatenate((v0_real.reshape(-1), v0_imag.reshape(-1), h0_real.reshape(-1), h0_imag.reshape(-1)))


extrema = []
avg_phase_diff = []
std_phase_diff = []
for i in tqdm(range(v_side)):
    phase_diff = []
    clamp = np.zeros((v_side, v_side))
    clamp[:int(np.ceil(v_side / 2.0)), i] = 1.0
    clamp[int(np.ceil(v_side / 2.0)):, v_side - i - 1] = 1.0
    for n in range(num_reps):
        z0 = init(v_side, h_side, i)
        cons = tuple([{'type' : 'eq',
		   'fun'  : constraint,
	 	   'args' : (j,)} for j in range(v_side**2)])
        res = minimize(Ham, z0, method='SLSQP', constraints=cons, options={'disp':False, 'maxiter':100})
        ex = res['x']
        ex_cplx = ex[:v_side**2].reshape(v_side, v_side) + 1j*ex[v_side**2:2*v_side**2].reshape(v_side, v_side)
        if sub_exp == 1:
            bar_1_avg_phase = np.mean(np.angle(ex_cplx[:int(np.ceil(v_side / 2.0)),i]))
            bar_2_avg_phase = np.mean(np.angle(ex_cplx[int(np.ceil(v_side / 2.0)):,v_side - i - 1]))
            phase_diff.append(np.abs(bar_1_avg_phase - bar_2_avg_phase))
        elif sub_exp == 2:
	    p = pool(ex)
	    phase_diff.append(np.abs(np.angle(p[0]) - np.angle(p[1])))
	elif sub_exp == 3:
           fig, ax = plt.subplots()
	   cplx_imshow(ax, ex_cplx, cm=plt.cm.hsv)
	   plt.savefig('/home/matt/geman_style_videos/e{0}.png'.format(i))
	   continue
	    
    avg_phase_diff.append(np.mean(phase_diff)) 
    std_phase_diff.append(np.std(phase_diff))
if sub_exp != 3:
    t = np.array(range(v_side)) - int(np.floor(v_side / 2.0))
    x = np.array(avg_phase_diff)
    s = np.array(std_phase_diff)
    plt.plot(t,x, color='#CC4F1B')
    plt.fill_between(t, x - s, x + s, edgecolor='#CC4F1B', facecolor='#FF9848', alpha=.5)
    plt.savefig('/home/matt/geman_style_videos/pool_phase_diff.png')

    
