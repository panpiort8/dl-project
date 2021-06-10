from einops import rearrange
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch

from universal_computation.datasets.dataset import Dataset
import numpy as np


class DigitAdditionDataset(Dataset):

    def __init__(self, batch_size, seq_length=None, min_v=0, max_v=9, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader
        self.seq_length = seq_length  # length of sequence
        self.min_v = min_v
        self.max_v = max_v


    def get_batch(self, batch_size=None, train=True):
        x = np.random.randint(self.min_v, self.max_v, size=(batch_size*self.seq_length))
        out = np.zeros(shape=(batch_size * self.seq_length, self.max_v - self.min_v + 1))
        
        out[range(x.shape[0]), x-self.min_v] = 1
        
        out = out.reshape((batch_size, self.seq_length, -1))
        
        y = x.reshape((batch_size, -1)).sum(axis=1).reshape((batch_size, 1))

        x = torch.tensor(out).to(device=self.device).float()
        y = torch.tensor(y).to(device=self.device).float()

#         print(x.shape, y.shape)
#         print(x, y)
#         print(x.dtype, y.dtype)


        return x, y