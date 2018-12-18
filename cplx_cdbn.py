from cplx_crbm import ComplexCRBM

class ComplexCDBN(object):

  def __init__(self):
    self.layers = []


  @classmethod
  def init_from_cdbn(cls, cdbn):
    me = cls()
    for crbm in cdbn:
      me.add_layer(ComplexCRBM.init_from_crbm(crbm))
    return me


  def _gibbs_step(self, t, vs, hs, ps, beta, beta_rate=crbm.CONST_ONE, max_beta=crbm.CONST_ONE, use_means=False):
    """INTENT: Start sampling from the second to last layer all the way to the visible,
               then back to the very deepest layer. So this goes back out and then in.
               This is the loop body used in `dbn_gibbs`' `while_loop`.
      Should handle nets with mixed pooling/nonpooling units.
    """
    t += 1
    beta = tf.minimum(beta * beta_rate, max_beta)

    # ***********************
    # For each hidden layer except the last one, sample the hidden/pooling
    # Going from deep to shallow.
    # this arr is temporary, we don't store states from the way down in the tensorarray
    way_down_hps = []
    for i in range(self.number_layer - 2, -1, -1):
      # Get layer below (V in Lee sec 3.6)
      if i == 0:
        below_p = vs.read(t - 1)
      else:
        below_p = ps[i - 1].read(t - 1)

      # Get layer above (H' in Lee sec 3.6)
      above_h = hs[i + 1].read(t - 1)

      topdown_signal = self.layer_name_to_object[self.layer_level_to_name[i + 1]].infer_probability(above_h, 'backward', beta=beta)

      cur_layer = self.layer_name_to_object[self.layer_level_to_name[i]]
      if cur_layer.prob_maxpooling:
        means = cur_layer.dbn_draw_samples(below_p, topdown_signal=topdown_signal, result='both', just_give_the_means=use_means, beta=beta)
      else:
        means = cur_layer.dbn_draw_samples(below_p, topdown_signal=topdown_signal, result='hidden', just_give_the_means=use_means, beta=beta)
      way_down_hps.insert(0, means)s

    # ************************
    # Now, sample the visible input.

    vis_h = means[0] if cur_layer.prob_maxpooling else means
    vis_layer = self.layer_name_to_object[self.layer_level_to_name[0]]
    vis_means = vis_layer.infer_probability(vis_h, 'backward', beta=beta)
    if use_means:
      vs = vs.write(t, vis_means)
    else:
      vs = vs.write(t, vis_layer.draw_samples(vis_means, method='backward'))

    # ************************
    # Finally, sample the hidden/pooling layers all the way in.

    # All but the last
    for i in range(0, self.number_layer - 1):
      # Get layer below (V in Lee sec 3.6)
      if i == 0:
        below_p = vis_means
      else:
        below_p = way_down_hps[i - 1]
        if self.layer_name_to_object[self.layer_level_to_name[i - 1]].prob_maxpooling:
          below_p = below_p[1]

      # Get layer above (H' in Lee sec 3.6)
      if i < self.number_layer - 2:
        above_h = way_down_hps[i + 1]
        if self.layer_name_to_object[self.layer_level_to_name[i + 1]].prob_maxpooling:
          above_h = above_h[0]
      else:
        above_h = hs[i + 1].read(t - 1)

      topdown_signal = self.layer_name_to_object[self.layer_level_to_name[i + 1]].infer_probability(above_h, 'backward', beta=beta)

      cur_layer = self.layer_name_to_object[self.layer_level_to_name[i]]
      if cur_layer.prob_maxpooling:
        h, p = cur_layer.dbn_draw_samples(below_p, topdown_signal=topdown_signal, result='both', just_give_the_means=use_means, beta=beta)
        hs[i] = hs[i].write(t, h)
        ps[i] = ps[i].write(t, p)
      else:
        h = p = cur_layer.dbn_draw_samples(below_p, topdown_signal=topdown_signal, result='hidden', just_give_the_means=use_means, beta=beta)
        hs[i] = hs[i].write(t, h)
        ps[i] = ps[i].write(t, p)

    # Last layer's hids/pools
    if self.fully_connected_layer == i + 1:
      p = tf.reshape(p, [self.batch_size, -1])
      p = tf.reshape(p, [self.batch_size, 1, 1, p.get_shape()[1].value])   

    last_layer = self.layer_name_to_object[self.layer_level_to_name[i + 1]]

    if last_layer.prob_maxpooling:
      last_means = last_layer.dbn_draw_samples(p, topdown_signal=None, result='both', just_give_the_means=use_means, beta=beta)
      hs[-1] = hs[-1].write(t, last_means[0])
      ps[-1] = ps[-1].write(t, last_means[1])
    else:
      last_means = last_layer.dbn_draw_samples(p, topdown_signal=None, result='hidden', just_give_the_means=use_means, beta=beta)
      hs[-1] = hs[-1].write(t, last_means)
      ps[-1] = ps[-1].write(t, last_means)

    return [t, vs, hs, ps, beta]



  def dbn_gibbs(self, start_vis_batch, n_gibbs, use_means=False, init_beta=crbm.CONST_ONE, max_beta=crbm.CONST_ONE, beta_rate=crbm.CONST_ONE):
    """INTENT: Gibbs sampling starting from a visible example.
               Should this be extended to start from something else?
               This just works with means right now, it never samples.

    start_vis_batch     numpy array!
    """
    beta = init_beta + 0.0
    input_placeholder = tf.placeholder(tf.float32, shape=self.input)

    # We'll store the visible and hidden/pooling layers at each timestep.
    vs = tf.TensorArray(tf.float32, size=n_gibbs + 1, clear_after_read=False, name='vs')
    hs = [tf.TensorArray(tf.float32, size=n_gibbs + 1, clear_after_read=False, name=('hs%d' % j)) for j in range(self.number_layer)]
    ps = [tf.TensorArray(tf.float32, size=n_gibbs + 1, clear_after_read=False, name=('ps%d' % j)) for j in range(self.number_layer)]

    # First, do a pass all the way into the network. This is just RBM style for now, not sure if it's
    # helpful here to have fantasies already in the deeper state.
    t = tf.constant(0)
    vs = vs.write(t, input_placeholder)
    ret_data = input_placeholder
    for i in range(self.number_layer):
      ret_layer = self.layer_name_to_object[self.layer_level_to_name[i]]
      if self.fully_connected_layer == i:
        ret_data = tf.reshape(ret_data, [self.batch_size, -1])
        ret_data = tf.reshape(ret_data, [self.batch_size, 1, 1, ret_data.get_shape()[1].value])   
      if ret_layer.prob_maxpooling:
        h, p = ret_layer.dbn_draw_samples(ret_data, topdown_signal=None, result='both', just_give_the_means=use_means, beta=beta)
        hs[i] = hs[i].write(t, h)
        ps[i] = ps[i].write(t, p)
        ret_data = p
      else:
        ret_data = ret_layer.dbn_draw_samples(ret_data, topdown_signal=None, result='hidden', just_give_the_means=use_means, beta=beta) 
        hs[i] = hs[i].write(t, ret_data)
        ps[i] = ps[i].write(t, ret_data)

    # Now, the while loop.
    # The loop body starts deep (second to last layer), comes to visible, and then goes deep (last layer) again.
    cond = lambda t, vs, hs, ps, b: tf.Print(t, [t], message="While loop step ") < n_gibbs
    loop_vars = [t, vs, hs, ps, beta]
    body = partial(self._gibbs_step, beta_rate=beta_rate, max_beta=max_beta, use_means=use_means)
    _, v_run, h_run, p_run, _ = tf.while_loop(cond, body, loop_vars)
  
    # Actually run the thing.
    return self.session.run(
      [v_run.stack(), [h.stack() for h in h_run], [p.stack() for p in p_run]],
      feed_dict={ input_placeholder: start_vis_batch.reshape(self.input) })

