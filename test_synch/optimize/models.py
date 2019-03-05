import torch
from torch import nn
import numpy as np
import ipdb

class deep(nn.Module):
    def __init__(self, img_side):
        super(deep, self).__init__()
	self.img_side = img_side
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4*4*64, 512)
        self.fc2 = nn.Linear(512, 10)


    def cplx_max_pool2d(self, z_norm, z_angle, k, stride):
        unfolded_z_norm   = z_norm.unfold(2, k, stride).unfold(3,k,stride)
	shp 	          = unfolded_z_norm.shape
	unfolded_z_norm   = unfolded_z_norm.reshape(shp[0], shp[1], shp[2], shp[3], -1)
	m, am             = torch.max(unfolded_z_norm,-1, keepdim=True)
	z_angle_unfolded  = z_angle.unfold(2,k,stride).unfold(3,k,stride).reshape(shp[0], shp[1], shp[2], shp[3], -1)
	am_angle          = torch.gather(z_angle_unfolded,-1, am)
	return (m*torch.cos(am_angle)).squeeze(-1), (m*torch.sin(am_angle)).squeeze(-1)
	

    def forward_cplx(self, z):
	z_real = z[:,0,:,:].unsqueeze(1)
	z_imag = z[:,1,:,:].unsqueeze(1)
	# Layer 1 
	batch_size = z_real.shape[0]
	z_real = self.conv1(z_real)
	z_imag = self.conv1(z_imag)
	z_norm = torch.sqrt((z_real)**2 + z_imag**2)
        z_norm = torch.sigmoid(z_norm)
	z_angle = torch.atan2(z_imag, z_real)
        z_real, z_imag = self.cplx_max_pool2d(z_norm, z_angle, 2, 2)

	#Layer 2
	z_real = self.conv2(z_real)
	z_imag = self.conv2(z_imag)
	z_norm = torch.sqrt((z_real)**2 + z_imag**2)
        z_norm = torch.sigmoid(z_norm)
	z_angle = torch.atan2(z_imag, z_real)
        z_real, z_imag = self.cplx_max_pool2d(z_norm, z_angle, 2, 2)

	# Layer 3
	z_real = z_real.view(-1,4*4*64)
	z_imag = z_imag.view(-1,4*4*64)
	z_real = self.fc1(z_real)
	z_imag = self.fc1(z_imag)
	z_norm = torch.sigmoid(torch.sqrt((z_real)**2 + z_imag**2))
	z_angle = torch.atan2(z_imag, z_real)
	z_real, z_imag = z_norm*torch.cos(z_angle), z_norm*torch.sin(z_angle)

	# Layer 4
	z_real, z_imag = self.fc2(z_real), self.fc2(z_imag)
	z_norm = torch.sqrt(z_real**2 + z_imag**2)
	z_norm = F.softmax(z_norm, dim=1)
	z_angle = torch.atan2(z_imag, z_real)
	
	return -1*torch.sqrt(torch.sum(z_norm**2)).mean()
	
class shallow(torch.nn.Module):
    def __init__(self,num_features, k=12):
	super(shallow, self).__init__()
	self.num_features = num_features
	self.k = k
	self.conv1 = torch.nn.Conv2d(1,self.num_features, (k, k), bias=False)
	torch.nn.init.xavier_uniform(self.conv1.weight)
    def forward(self,x):
	x_real = x[:,0,:,:].unsqueeze(1)
	x_imag = x[:,1,:,:].unsqueeze(1)

        x_real = self.conv1(x_real)
	x_imag = self.conv1(x_imag)

	# All hidden unit lengths. Norms in R2
	x_norm_sq = x_real**2 + x_imag**2
 	# Layer magniude. Norms in R^(2*k^2) 
        return -1*(torch.sum(x_norm_sq**2, dim=(2,3)) / float(self.k**2)).mean()
	
class shallow_real(torch.nn.Module):
    def __init__(self,num_features, k=12):
	super(shallow_real, self).__init__()
	self.num_features = num_features
	self.k = k
	self.conv1 = torch.nn.Conv2d(1,self.num_features, (k, k), bias=False)
	
	torch.nn.init.xavier_uniform(self.conv1.weight)
    def forward(self,x):
	x = self.conv1(x)

	# All hidden unit lengths. Norms in R2
 	# Layer magniude. Norms in R^(2*k^2) 
        return -1*(torch.sum(x**2, dim=(2,3)) / float(self.k**2)).mean()
