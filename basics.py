import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

# Create Tensor
x = Variable(torch.randn(5, 3))
y = Variable(torch.randn(5, 2))

# Build a linear layer
linear = nn.Linear(3, 2)

print ('w: ', linear.weight)
print ('b: ', linear.bias)

critic = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# forward propagate