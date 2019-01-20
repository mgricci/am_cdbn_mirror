import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import numpy as np
import ipdb
from scipy.optimize import minimize
import itertools
from sympy.utilities.iterables import multiset_permutations as mp

num_hid = 4
num_vis = 4
#w = np.random.normal(0,1,size=(num_hid*num_vis))

#w = np.array([1,-1,-1,-1, 1, -1, -1, -1, 1])
#w = -1*np.ones((num_hid, num_vis))
#mask= -1*np.eye(num_hid,  num_vis)
#mask[mask==0] = 1
#w*= mask
#w = w.reshape(num_hid*num_vis)
#w = -1*np.array([1,-1,-1,-1, 1, -1, -1, -1, 1])
#w  = np.ones(9)
w = np.array([10000,-1,-1,-1])
w = np.array(list(mp(w)))
w = w.reshape(num_vis*num_hid)
def Ham(x):
    
    hidden  = x[:num_hid]
    visible = x[num_hid:]
    pairs = np.array([[i,j] for i in hidden for j in visible])
    return -1* np.dot(w,np.cos(pairs[:,0] - pairs[:,1]))

fixed_points = []
n_samples = 1000

for i in range(n_samples):
    x_0 = 2*np.pi*np.random.rand(num_hid + num_vis)
    res = minimize(Ham, x_0, method='nelder-mead', options={'xtol': 1e-12, 'disp': False})
    fixed_points.append(res.x)

    #print('Hidden: ' + str(res.x[:num_hid] % (2*np.pi)))
    #print('Visible: ' + str(res.x[num_hid:] % (2*np.pi)))

    #for v, vis in enumerate(res.x[num_hid:]): 
    #    for w, wis in enumerate(res.x[num_hid:]): 
    #        if v != w: 
    #	        print('Difference between unit {} and unit {} is {}'.format(v, w, (vis - wis) % (2*np.pi))) 
fixed_points = np.array(fixed_points)
for l in ['v', 'h']:
    fig, ax = plt.subplots()
    for i in range(n_samples):
        fp = fixed_points[i,num_hid:] if l == 'v' else fixed_points[i, :num_hid]
        mn = np.min(fp)
        rot_fp = fp - mn
        units = np.exp(1j*rot_fp)
        x = np.real(units) + .05*np.random.normal(0,.05, size=units.shape)
        y = np.imag(units) + 0.5*np.random.normal(0, .05, size=units.shape)
        plt.scatter(x, y, color='#CC4F1B', alpha=.25)

    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    circ = matplotlib.patches.Circle((0,0), radius=1, facecolor=None, fill=False, edgecolor='k')
    ax.add_artist(circ)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('{}_circle.png'.format(l))
    plt.close()
print('done')

