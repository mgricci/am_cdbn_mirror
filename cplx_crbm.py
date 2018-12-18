from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.special


# *************************************************************************** #
# Some complex library functions. Just gonna put these here for now.

def as_cplx(tensor):
  if tensor.dtype.is_complex:
    return tensor
  else:
    return tf.complex(tensor, tf.constant(0.0, tensor.dtype))

def complex_conv2d(tensor1, tensor2, strides, padding='VALID'):
    real1, imag1 = tf.real(tensor1), tf.imag(tensor1)
    real2, imag2 = tf.real(tensor2), tf.imag(tensor2)
    conv_real = ( tf.nn.conv2d(real1, real2, strides, padding=padding)
                - tf.nn.conv2d(imag1, imag2, strides, padding=padding))
    conv_imag = ( tf.nn.conv2d(real1, imag2, strides, padding=padding)
                + tf.nn.conv2d(imag1, real2, strides, padding=padding))
    return tf.complex(conv_real, conv_imag)

def bessel_i0(x, dtype=tf.float64):
  if dtype != x.dtype:
    # the 'need more bits' version, which is the default for our usual float32s.
    return tf.cast(
      tf.py_func(scipy.special.i0, [tf.cast(x, dtype)], dtype),
      x.dtype)
  else:
    return tf.py_func(scipy.special.i0, [x], dtype)

def phasor(tensor):
  return tf.exp(1j * as_complex(tf.angle(tensor)))

def vonmises(mu, kappa):
  samples = tf.py_func(np.random.vonmises, [mu, kappa], tf.float64)
  return tf.cast(samples, dtype=tf.float32)

# *************************************************************************** #

class ComplexCRBM(object):

  def __init__(self, name, fully_connected, v_height, v_width, v_channels,
      f_height, f_width, f_number, prob_maxpooling, padding, batch_size,
      real_kernels):
    self.name = name
    self.fully_connected = fully_connected
    self.v_height = v_height
    self.v_width = v_width
    self.v_channels = v_channels
    self.f_height = f_height
    self.f_width = f_width
    self.f_number = f_number
    self.prob_maxpooling = prob_maxpooling
    self.padding = padding
    self.batch_size = batch_size
    if padding:
      self.hidden_height       = v_height
      self.hidden_width        = v_width
    else:
      self.hidden_height       = v_height - f_height + 1
      self.hidden_width        = v_width - f_width + 1

    # Get params
    self.kernels = as_cplx(real_kernels)
    # TODO: Not sure what to do with the biases. Gonna leave em alone for now.


  @classmethod
  def init_from_crbm(cls, crbm):
    if crbm.gaussian_unit:
      raise(tf.errors.InvalidArgumentError(
        "Not sure how to complex-ify a Gaussian-Bernoulli RBM."))

    return cls('cplx_' + crbm.name, crbm.fully_connected, crbm.v_height,
      crbm.v_width, crbm.v_channels, crbm.f_height, crbm.f_width,
      crbm.f_number, crbm.prob_maxpooling, crbm.padding, crbm.batch_size,
      crbm.kernels)


  # ************************************************************************* #
  # The following methods are verbatim from real RBM.
  # Using their style for these things to keep this translation simple.

  def _get_flipped_kernel(self):
    return tf.transpose(
      tf.reverse(self.kernels, [0, 1]),
      perm=[0, 1, 3, 2])
      
        
  def _get_padded_hidden(self, hidden):
    return tf.pad(
      hidden,
      [[0, 0], 
       [self.filter_height-1, self.filter_height-1],
       [self.filter_width-1, self.filter_width-1],
       [0, 0]]) 
      
        
  def _get_padded_visible(self, visible):
    return tf.pad(
      visible,
      [[0, 0],
       [np.floor((self.filter_height-1)/2).astype(int),
        np.ceil((self.filter_height-1)/2).astype(int)],
       [np.floor((self.filter_width-1)/2).astype(int),
        np.ceil((self.filter_width-1)/2).astype(int)],
       [0, 0]])


  # ************************************************************************* #
  # The rest of this class holds the minimal translation of the real RBM's
  # logic to the complex setting. I'm only translating the functions that
  # are necessary for Gibbs sampling (i.e. the ones called in the
  # `ComplexCRBM.{dbn_gibbs,_gibbs_step}` methods.

  def infer_probability(self, operand, method, result='hidden', beta=CONST_ONE):
    '''
    This is a misnomer in this net, since we are computing parameters
    for bernoulli and vonMises, not just probabilities. Keeping the name for
    consistency.
    RETURNS: please look at the return statement for the arguments you supplied,
    because they're all different.
    '''
    if method == 'forward': 
      # Propagate
      if self.padding:
        conv = complex_conv2d(operand, self.kernels, [1, 1, 1, 1], padding='SAME')
      else:
        conv = complex_conv2d(operand, self.kernels, [1, 1, 1, 1], padding='VALID')

      # Compute means
      if self.prob_maxpooling: 
        # Rate probs in probmaxpool
        a = tf.abs(conv)
        alpha = tf.angle(conv)
        bessels = bessel_i0(a)
        custom_kernel = tf.constant(1.0, shape=[2,2,self.filter_number,1])
        sum_bessels = tf.nn.depthwise_conv2d(bessels, custom_kernel, [1, 2, 2, 1], padding='VALID')
        bessel_sftmx_denom = tf.add(1.0,sum_bessels)
        ret_kernel = np.zeros((2,2,self.filter_number,self.filter_number))
        for i in range(2):
          for j in range(2):
            for k in range(self.filter_number):
              ret_kernel[i,j,k,k] = 1
        custom_kernel_bis = tf.constant(ret_kernel,dtype = tf.float32)
        upsampled_denom = tf.nn.conv2d_transpose(bessel_sftmx_denom, custom_kernel_bis,
          (self.batch_size,self.hidden_height,self.hidden_width,self.filter_number),
          strides= [1, 2, 2, 1], padding='VALID', name=None)
        hid_cat_P = tf.div(bessels, upsampled_denom)
        pool_P = tf.subtract(1.0, tf.div(1.0, bessel_sftmx_denom))
        # Now get von Mises params for hids.
        # We're not gonna give vM params for pools. It's not vM so dwai.
        hid_mu = alpha
        # TODO: whoops. this is wrong. it should be samples from rates, not ber_P. will
        # need to do some more serious restructuring to fix this.
        hid_kappa = a * hid_cat_P
        if result == 'hidden':
          return hid_cat_P, hid_mu, hid_kappa
        elif result == 'pooling': 
          raise(tf.errors.InvalidArgumentError("Can't just sample pools in cplx case."))
        elif result == 'both':
          return hid_cat_P, pool_P, hid_mu, hid_kappa
      else:
        # Rate probs no probmaxpool
        b = bessel_i0(conv)
        hid_P = b / (1.0 + b)
        # hid_P, hid_mu, hid_kappa.
        return hid_P, alpha, a * hid_P
    
    'Computing VISIBLE layer with HIDDEN layer given'
    if method == 'backward':
      if self.padding:
        conv = complex_conv2d(operand, self._get_flipped_kernel(), [1, 1, 1, 1], padding='SAME')
      else:
        conv = complex_conv2d(self._get_padded_hidden(operand),
          self._get_flipped_kernel(), [1, 1, 1, 1], padding='VALID')
      a = tf.abs(conv)
      alpha = tf.angle(conv)
      b = bessel_i0(conv)
      vis_P = b / (1.0 + b)
      return vis_P, alpha, a * vis_P



  def dbn_infer_probability(self, my_visible, topdown_signal=None, result='hidden', beta=CONST_ONE):
    """INTENT : Compute the probabily of activation of the hidden or pooling layer given the visible aka prev pooling,
                and the next layer's hiddens (not poolings!)
    """
    raise ValueError("Still need to translate to complex.")

    if topdown_signal is None:
      return self.infer_probability(my_visible, 'forward', result=result)

    'Computing HIDDEN layer with MY VISIBLE, NEXT HIDDEN layers given'
    'Gaussian visible or not, hidden layer activation is a sigmoid'
    # handle the problem case where the next layer is fc. the impl makes weird shapes.
    if self.padding:
      conv = tf.nn.conv2d(my_visible, self.kernels, [1, 1, 1, 1], padding='SAME')
    else:
      conv = tf.nn.conv2d(my_visible, self.kernels, [1, 1, 1, 1], padding='VALID')
    bias = beta * tf.nn.bias_add(conv, self.biases_H)
    if self.prob_maxpooling: 
      'SPECIFIC CASE where we enable probabilistic max pooling'
      'This is section 3.6 in Lee!'
      topdown_signal = tf.reshape(topdown_signal, (self.batch_size, self.hidden_height // 2, self.hidden_width // 2, self.filter_number))
      custom_kernel = tf.constant(1.0, shape=[2,2,self.filter_number,1])
      # sum += topdown_signal
      # sum = tf.add(1.0,sum)
      ret_kernel = np.zeros((2,2,self.filter_number,self.filter_number))
      for i in range(2):
        for j in range(2):
          for k in range(self.filter_number):
            ret_kernel[i,j,k,k] = 1
      custom_kernel_bis = tf.constant(ret_kernel,dtype = tf.float32)
      supersampled_topdown = tf.nn.conv2d_transpose(topdown_signal, custom_kernel_bis,
        (self.batch_size,self.hidden_height,self.hidden_width,self.filter_number), strides= [1, 2, 2, 1], padding='VALID')
      total_hidshape_signal = supersampled_topdown + bias
      exp = tf.exp(total_hidshape_signal)
      poolshape_denom = 1 + tf.nn.depthwise_conv2d(exp, custom_kernel, [1, 2, 2, 1], padding='VALID')
      hidshape_denom = tf.nn.conv2d_transpose(poolshape_denom, custom_kernel_bis,
        (self.batch_size,self.hidden_height,self.hidden_width,self.filter_number), strides= [1, 2, 2, 1], padding='VALID')
      if result == 'hidden': 
        'We want to obtain HIDDEN layer configuration'
        return tf.div(exp, hidshape_denom)
      elif result == 'pooling': 
        'We want to obtain POOLING layer configuration'
        return tf.subtract(1.0,tf.div(1.0, poolshape_denom))
      elif result == 'both':
        return (tf.div(exp, hidshape_denom), tf.subtract(1.0,tf.div(1.0, poolshape_denom)))

    topdown_signal = tf.reshape(topdown_signal, (self.batch_size, self.hidden_height, self.hidden_width, self.filter_number))
    return tf.sigmoid(bias + topdown_signal)

      
        
        
        

  def draw_samples(self, mean_activation, method='forward', beta=CONST_ONE):
    """INTENT : Draw samples from distribution of specified parameter
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS :
    mean_activation         :        parameter of the distribution to draw sampels from
    method                  :        which direction for drawing sample ie forward or backward
    ------------------------------------------------------------------------------------------------------------------------------------------
    REMARK : If FORWARD then samples for HIDDEN layer (BERNOULLI)
             If BACKWARD then samples for VISIBLE layer (BERNOULLI OR GAUSSIAN if self.gaussian_unit = True)"""
    raise ValueError("Still need to translate to complex.")

    if method == 'forward':
      height   =  self.hidden_height
      width    =  self.hidden_width
      channels =  self.filter_number  
    elif method == 'backward':
      height   =  self.visible_height
      width    =  self.visible_width
      channels =  self.visible_channels
    return tf.where(tf.random_uniform([self.batch_size,height,width,channels]) - mean_activation < 0, tf.ones([self.batch_size,height,width,channels]), tf.zeros([self.batch_size,height,width,channels]))






  @staticmethod
  def _dbn_maxpool_sample_helper(hid_probs, pool_probs):
    """ Helper for multinomial sampling. """
    # Store shapes for later
    hs, ps = hid_probs.shape, pool_probs.shape

    # First, we extract blocks and their corresponding pooling unit
    # We need to use doubles because of precision problems in np.random.multinomial. Wow did this take me a while to figure out. https://github.com/numpy/numpy/issues/8317
    hid_prob_patches = np.asarray([
        imx.view_as_blocks(im, (2, 2, 1)) for im in hid_probs
      ]).reshape(-1, 4).astype(np.float64)
    pool_probs_for_patches = np.asarray([
        imx.view_as_blocks(im, (1, 1, 1)) for im in pool_probs
      ]).reshape(-1, 1)

    # hid_prob_patches will not sum to 1 over each block, which it needs to for the multinomial sampler
    # but sometimes we don't even want to sample any hid in a block (i.e. when pooling unit is 0), deal w that later.
    # first, normalize hidprobs and sample.
    psums = hid_prob_patches.sum(axis=1)
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


  def dbn_draw_samples(self, my_visible, topdown_signal=None, result='hidden', beta=CONST_ONE, just_give_the_means=False):      
    """ This correctly samples the hids/pools like in Lee, so that only one hid per block is active.
        In a maxpool, this is a py_func situation right now. sorry. """
    raise ValueError("Still need to translate to complex.")
    if just_give_the_means:
      return self.dbn_infer_probability(my_visible, topdown_signal=topdown_signal, result=result, beta=beta)

    if self.prob_maxpooling:
      hid_probs, pool_probs = self.dbn_infer_probability(my_visible, topdown_signal=topdown_signal, result='both', beta=beta)
      hid_samples, pool_samples = tf.py_func(
        self._dbn_maxpool_sample_helper,
        [hid_probs, pool_probs],
        (tf.float32, tf.float32))
      hid_samples.set_shape([self.batch_size, self.hidden_height, self.hidden_width, self.filter_number])
      pool_samples.set_shape([self.batch_size, self.hidden_height // 2, self.hidden_width // 2, self.filter_number])
      if result == 'hidden':
        return hid_samples
      if result == 'both':
        return (hid_samples, pool_samples)
      elif result == 'pooling':
        return pool_samples
    elif result == 'hidden':
      # No pooling, standard sampler will suffice, just need to use dbn probs to incorporate layer above.
      return self.draw_samples(self.dbn_infer_probability(my_visible, topdown_signal=topdown_signal, beta=beta))
    else:
      raise ValueError("Make sure result makes sense w this layer type.")





