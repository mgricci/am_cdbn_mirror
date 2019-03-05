import torch 
import numpy as np
from torch.utils.data import Dataset

class bars(Dataset):
    def __init__(self,
		 img_side=32,
		 num_bars=6):
        super(bars, self).__init__()
	self.img_side=img_side
	self.num_bars=num_bars
    def __len__(self):
        return 99999999
    def __getitem__(self, index):
        img = torch.zeros((1, self.img_side, self.img_side))
        rand_v_locs = torch.randperm(self.img_side)[:self.num_bars]
        rand_h_locs = torch.randperm(self.img_side)[:self.num_bars]
        img[:,:,rand_v_locs] = 1.0
	img[:,rand_h_locs,:] = 1.0
        return (img,0)

