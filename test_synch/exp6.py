import numpy as np
from numpy.random import vonmises as vm
from scipy.special import expit as sigmoid
from scipy.special import i0 as bessel
from scipy.signal import convolve2d as conv
import skimage.util as imx
from test_utils import save_cplx_anim, generate_shape_img, generate_bars_img
import ipdb
# SHAPES BARS EXPERIMENT:

class my_net(object):
    def __init__(self, in_shape=17,
		       weights = [1.0],
		       kernel_size = 5,
		       conv_stride = 5,
		       pool_size = 2,
		       n_gibbs = 32,
		       sigma = 1.0, 
		       bias = 0.0):
        # set attributes, including sampling type
	self.in_shape    = in_shape
	self.n_gibbs     = n_gibbs
	self.pool_size   = pool_size 
	self.pool_stride = pool_size
	self.conv_stride = conv_stride
	self.kernel_size = kernel_size
	# Filters
	self.make_filters(kernel_size, weights[0])
	self.hid_shape = int(np.ceil((self.in_shape - self.filters[0].shape[0] + 1) / float(self.conv_stride)))
	self.pool_shape = int(np.ceil(self.hid_shape / float(pool_size)))
	self.biases = bias*np.ones((self.hid_shape, self.hid_shape))
	self.sigma_sq = sigma**2
	#self.test_filters()
    def test_filters(self):
        norm_w = np.sqrt(np.sum(self.filters[0]**2))
	msq    = self.kernel_size**2
	in_range = (norm_w * msq <= self.sigma_sq*1.93)
	
	print('Maximum drive: {} \n Excitability threshold: {}'.format(norm_w*msq, self.sigma_sq*1.93))
	if in_range:
	    print('The network is tonically inactive with minimum firing probability {}'.format(self.sigma_sq *.5))
	else:
	    print('The network is tonically excited with minimum firing probability {}'.format(self.sigma_sq *.5))
    def make_filters(self, kernel_size=5, weight=1.0):
        filter1 = -weight*np.ones((kernel_size, kernel_size))
	#filter1 = np.zeros((kernel_size, kernel_size))
	mid = int(np.floor(kernel_size / 2.))
	filter1[:,mid] *= -1
	#filter1 -= np.mean(filter1)
	self.filters = [filter1]
        self.adjoint_filters = [np.flip(f, (0,1)) for f in self.filters]
		
    def _dbn_maxpool_sample_helper(self, hid_probs, pool_probs):
        """ Helper for multinomial sampling. Just the rates. """
        # Store shapes for later
	hs, ps = hid_probs.shape, pool_probs.shape

        # First, we extract blocks and their corresponding pooling unit
	# We need to use doubles because of precision problems in np.random.multinomial. Wow did this take me a while to figure out. https://github.com/numpy/numpy/issues/8317
	hid_prob_patches = imx.view_as_blocks(hid_probs, (self.pool_size,self.pool_size)).reshape(-1,self.pool_size**2)
	pool_probs_for_patches = imx.view_as_blocks(pool_probs, (1,1)).reshape(-1,1)
	#hid_prob_patches = np.asarray([
	#    imx.view_as_blocks(im, (2, 2)) for im in hid_probs
	#    ]).reshape(-1, 4).astype(np.float64)
	#pool_probs_for_patches = np.asarray([
	#    imx.view_as_blocks(im, (1, 1, 1)) for im in pool_probs
	#    ]).reshape(-1, 1)

	# hid_prob_patches will not sum to 1 over each block, which it needs to for the multinomial sampler
	# but sometimes we don't even want to sample any hid in a block (i.e. when pooling unit is 0), deal w that later.
	# first, normalize hidprobs and sample.
	psums = hid_prob_patches.sum(axis=1)
	#print('x',np.isfinite(psums).all())
	# thing is the total prob could be 0 over some patches, when p(pool=0)=1. have to deal with that
	hid_prob_patches[psums <= 1e-8, ...] = 0.25
    	psums[psums <= 1e-8] = 1
	hid_prob_patches /= psums[..., None]
	patch_hid_samples = np.zeros(hid_prob_patches.shape, dtype=np.float32)
	for i in range(hid_prob_patches.shape[0]):
	  try:
	      patch_hid_samples[i] = np.random.multinomial(1, hid_prob_patches[i])
	  except:
	      print(hid_prob_patches[i].dtype, hid_prob_patches[i].sum(), hid_prob_patches[i])
	      raise
			
	# sample pools
	patch_pool_samples = np.random.binomial(1, pool_probs_for_patches).astype(np.float32)
	# set hids to 0 when pooling unit is off
	patch_hid_samples *= patch_pool_samples
	# reshape and return.
	return patch_hid_samples.reshape(hs), patch_pool_samples.reshape(ps)

    def windolf_sampling(self, params, layer):
	a = np.abs(params)
	alpha = np.angle(params)

	if layer=='visible':
	    rates = np.abs(self.clamp)
	    phases = vm(alpha, a*rates / self.sigma_sq)
	    return rates*np.exp(1j*phases)
	else:
	    bessels = bessel(a - self.biases)/ self.sigma_sq
	    custom_kernel = np.ones((self.pool_size, self.pool_size))
	    sum_bessels = conv(bessels, custom_kernel, mode='valid')
	    # Downsample
	    sum_bessels = sum_bessels[0::self.pool_stride, 0::self.pool_stride]
	    
	    bessel_sftmx_denom = 1.0 + sum_bessels
	    upsampled_denom = bessel_sftmx_denom.repeat(self.pool_stride, axis=0).repeat(self.pool_stride, axis=1)    
	    hid_cat_P = bessels / upsampled_denom
	    pool_P = 1.0 - 1.0 / bessel_sftmx_denom
		
	    hid_rates, pool_rates = self._dbn_maxpool_sample_helper(hid_cat_P, pool_P)
	    hid_phases = vm(alpha, a*hid_rates / self.sigma_sq)
	    hid_samples = hid_rates*np.exp(1j*hid_phases)
	    pool_phases = np.sum(imx.view_as_blocks(hid_phases, (self.pool_size, self.pool_size)), axis=(2,3))
	    pool_samples = pool_rates*np.exp(1j*pool_phases)
	    return hid_samples, pool_samples
	

    def x_in(self, vis):
        return conv(vis,self.filters[0], mode='valid')[::self.conv_stride, ::self.conv_stride]

    def x_out(self, hid):
	ups = np.zeros((self.in_shape - self.filters[0].shape[0] + 1, self.in_shape - self.filters[0].shape[1] + 1))
	ups[::self.conv_stride, ::self.conv_stride] = hid
	hid = np.pad(hid, ((self.filters[0].shape[0] - 1, self.filters[0].shape[0] - 1), (self.filters[0].shape[1] - 1, self.filters[0].shape[1] - 1)), mode='constant')
	return conv(hid, self.adjoint_filters[0], mode='valid')

    def run(self, clamped_visible):
	self.clamp = clamped_visible
	self.v_run = [clamped_visible*np.exp(1j*2*np.pi*np.random.rand(self.in_shape, self.in_shape))]
	self.h_run = [np.zeros((self.hid_shape, self.hid_shape))]
	self.p_run = [np.zeros((self.pool_shape, self.pool_shape))]
        for i in range(self.n_gibbs):
	    print('Iteration {}'.format(i))
	    hidden_activations = self.x_in(self.v_run[i-1])
	    hidden_samples, pool_samples = self.windolf_sampling(hidden_activations, 'hidden')
            self.h_run.append(hidden_samples)
	    self.p_run.append(pool_samples)
	    visible_activations = self.x_out(self.h_run[i+1])
	    visible_samples = self.windolf_sampling(visible_activations, 'visible')
	    self.v_run.append(visible_samples)

        return self.v_run, self.h_run, self.p_run
kernel_size = 4
im_size = 11
pad_size = 2
in_shape = im_size + 2*pad_size
net = my_net(in_shape=in_shape, kernel_size=kernel_size,weights=[.01],conv_stride=1, pool_size=2, n_gibbs=32, sigma=.5, bias=0.0)
fo i in range(10):
    stim = np.zeros((im_size, im_size))
    stim[:int(im_size / 2.0), i] = 1.0
    stim[int(im_size / 2.0):, -1*(i+1)] = 1.0
    stim = np.pad(stim, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
    #stim[:,4] = 1.0
    v, h, p = net.run(stim)
    v, h, p = np.array(v), np.array(h), np.array(p)
    v = np.expand_dims(np.expand_dims(v,0),-1)
    h = np.expand_dims(np.expand_dims(h,0),-1)
    p = np.expand_dims(np.expand_dims(p,0),-1)
    save_cplx_anim('/home/matt/geman_style_videos/v{}'.format(i), v)
    save_cplx_anim('/home/matt/geman_style_videos/h{}'.format(i),h, number=1)
    save_cplx_anim('/home/matt/geman_style_videos/p{}'.format(i),p)
print('done')
