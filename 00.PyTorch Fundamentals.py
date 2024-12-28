## 00. PyTorch Fundamentals
# Importing pyTorch Library
import torch
torch.__version__

# Scalars
scalar = torch.tensor(7)
scalar
#Checking the dimension of scalar
scalar.ndim ##Scalar doesnt have any dimension, so the output will be 0

# Vector

vector = torch.tensor([7, 7])
vector
#Checking the dimension of vector
vector.ndim ## Vector has 1 dimension
vector.shape ## checking the shape of vector 