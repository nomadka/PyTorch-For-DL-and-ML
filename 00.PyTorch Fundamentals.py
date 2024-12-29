# 00. PyTorch Fundamentals
# Importing pyTorch Library
import torch
torch.__version__

# Scalars
scalar = torch.tensor(7)
scalar
#Checking the dimension of scalar
scalar.ndim ##Scalar doesnt have any dimension, so the output will be 0

#Vector
vector = torch.tensor([7, 7])
vector
#Checking the dimension of vector
vector.ndim ## Vector has 1 dimension
vector.shape ## checking the shape of vector 
#Matrix
matrix = torch.tensor([[6, 2],
                      [8, 2]])
matrix
#Checking the dimension of matrix
matrix.ndim ## Matrix has 2 dimension
matrix.shape ## checking the shape of matrix

#Tensors
TENSOR = torch.tensor([[
                       [[1,2,3,9],
                        [2,4,5,9],
                        [3,6,7,9],
                        [7,8,9,9]]]])
TENSOR
TENSOR.ndim ## Tensor has 3 dimension
TENSOR.shape ## checking the shape of Tensor
TENSOR[0,1]

#Random Tensors
random_tensor = torch.rand(2,3,2)
random_tensor
random_image_size_tensor = torch.rand(size=(224,224,3)) # height, width, color channel
random_image_size_tensor.shape, random_image_size_tensor.ndim