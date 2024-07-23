import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import importlib
from Quanvolution import QuanvLayer1D
import numpy as np
torch.manual_seed(0)
np.random.seed(0)  

class CNN(nn.Module):
  '''
    Convolutional Neural Network
  '''
  def __init__(self,kernel_size,n_layers,ansatz,out_size,out_channels,architecture):
    super().__init__()
    self.out_size = out_size
    self.layers = nn.Sequential(
      QuanvLayer1D(out_channels, kernel_size,  n_layers ,ansatz,architecture),
      
      nn.ReLU(),
      nn.MaxPool1d(out_size), # We compute the maximum of each channel = 1D global max-pooling.
      nn.Flatten(),
      nn.Linear(out_channels, 1),
      
    )


  def forward(self, x):
    # In the Qconv1D layer, we have input size = N, C_in, L, where N is the batch_size,
    # C_in is the number of input channels and L is a length of signal sequence. 
    x = x.view(-1,1,self.out_size) # We have only 1 vector therefore we have only 1 channel.
    '''Forward pass'''
    return self.layers(x)