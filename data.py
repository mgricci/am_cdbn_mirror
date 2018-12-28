from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import ipdb

""" --------------------------------------------
    ------------------- DATA -------------------
    -------------------------------------------- """
	
class MNIST_HANDLER(object):
  
  def __init__(self, data):
    self.num_training_example = data.train.num_examples
    self.num_test_example     = data.test.num_examples
    self.training_data , self.training_labels = data.train.next_batch(self.num_training_example)
    self.test_data , self.test_labels        = data.test.next_batch(self.num_test_example)
    self.whiten               = False
    self.training_index       = -20
    self.test_index           = -20
    
    
  def do_whiten(self):
    self.whiten         = True
    data_to_be_whitened = np.copy(self.training_data)
    mean                = np.sum(data_to_be_whitened, axis = 0)/self.num_training_example
    mean                = np.tile(mean,self.num_training_example)
    mean                = np.reshape(mean,(self.num_training_example,784))
    centered_data       = data_to_be_whitened - mean                
    covariance          = np.dot(centered_data.T,centered_data)/self.num_training_example
    U,S,V               = np.linalg.svd(covariance)
    epsilon = 1e-5
    lambda_square       = np.diag(1./np.sqrt(S+epsilon))
    self.whitening_mat  = np.dot(np.dot(U, lambda_square), V)    
    self.whitened_training_data  = np.dot(centered_data,self.whitening_mat)
    
    data_to_be_whitened = np.copy(self.test_data)
    mean                = np.sum(data_to_be_whitened, axis = 0)/self.num_test_example
    mean                = np.tile(mean,self.num_test_example)
    mean                = np.reshape(mean,(self.num_test_example,784))
    centered_data       = data_to_be_whitened - mean  
    self.whitened_test_data  = np.dot(centered_data,self.whitening_mat)

    
  def next_batch(self, batch_size, type = 'train'):
    if type == 'train':
      if self.whiten:
        operand = self.whitened_training_data
      else:
        operand = self.training_data
      operand_bis = self.training_labels
      self.training_index = (batch_size + self.training_index) % self.num_training_example
      index = self.training_index
      number = self.num_training_example
    elif type == 'test':
      if self.whiten:
        operand = self.whitened_test_data
      else:
        operand = self.test_data
      operand_bis = self.test_labels
      self.test_index = (batch_size + self.test_index) % self.num_test_example
      index = self.test_index
      number = self.num_test_example

    if index + batch_size > number:
      part1 = operand[index:,:]
      part2 = operand[:(index + batch_size)% number,:]
      result = np.concatenate([part1, part2])
      part1 = operand_bis[index:,:]
      part2 = operand_bis[:(index + batch_size)% number,:]
      result_bis = np.concatenate([part1, part2])
    else:
      result = operand[index:index + batch_size,:]
      result_bis = operand_bis[index:index + batch_size,:]
    return result, result_bis
	
class SHAPE_HANDLER(object):
  
  def __init__(self, height=20, width=20, pad_width=0):
    self.height               = height
    self.width                = width
    self.pad_width            = pad_width
    
  def generate_shape_img(self, width, height, nr_shapes=3, pad_width = 0):

    square = np.array(
    [[1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1]])

    triangle = np.array(
    [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    shapes = [square, triangle, triangle[::-1, :].copy()]
    img = np.zeros((height, width))

    for i in range(nr_shapes):
        shape = shapes[np.random.randint(0, len(shapes))]
        sy  , sx = shape.shape
        x = np.random.randint(0, width-sx+1)
        y = np.random.randint(0, height-sy+1)
        region = (slice(y,y+sy), slice(x,x+sx))
        img[region][shape != 0] += 1
	img[img>1] = 1
    return np.expand_dims(np.pad(img, pad_width, 'constant'), -1)
	
  def next_batch(self, batch_size, type=None):
    '''
        Shapes generator
    '''
    height = self.height
    width  = self.width
    
    return np.array([self.generate_shape_img(height, width, pad_width=self.pad_width) for b in range(batch_size)]).astype(np.float32), None
