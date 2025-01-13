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

#Random Tensors -> Why = Start with random numbers -> Look at data -> Update random number -> look at data 
random_tensor = torch.rand(2,3,2)
random_tensor
# Random tensor with similar image shape
random_image_size_tensor = torch.rand(size=(224,224,3)) # height, width, color channel
random_image_size_tensor.shape, random_image_size_tensor.ndim

# Tensors with all Zeros
zeros = torch.zeros(size=(3,5))
zeros
# Create a tensor of all ones
ones = torch.ones(size=(3,4))
ones
# Operations in tensor
ones_plus_one = ones + 1
ones_plus_one
ones_mul_two = ones * 2
ones_mul_two

# Creating range of tensors
one_to_ten = torch.arange(start=1,end=20,step=2)
one_to_ten

# Creating tensors like
ten_zeros = torch.zeros_like(input = one_to_ten)
ten_zeros

# Tensor Data Type
# Float 32 Tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], 
                                dtype=None, # Data type of the tensor e.g float32, floar16 
                                device=None, # In which device your tensor is on
                                  requires_grad=False) # wheather or not to track gradients with this tensor operation
float_32_tensor
float_32_tensor.dtype # By default the pyTorch have float 32 datatype

float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor.dtype
float_16_tensor * float_32_tensor
float_32_tensor.ndim
int_32_tensor = torch.tensor([5,4,3], dtype=torch.int32)
int_32_tensor.dtype
float_32_tensor*int_32_tensor

# Getting Information from Tensor
some_tensor = torch.rand(3,4)
some_tensor
print(some_tensor)
print(f"Data type of Tensor: {some_tensor.dtype}")
print(f"Shape of Tensor: {some_tensor.shape}" )
print(f"Device of Tensor: {some_tensor.device}")
x_tensor = torch.ones(size=(2,3),dtype=torch.int16)
print(x_tensor)
print(f"Data type of Tensor: {x_tensor.dtype}")
print(f"Shape of Tensor: {x_tensor.shape}" )
print(f"Device of Tensor: {x_tensor.device}")

##Manipulating Tensors (Tensor Operations +-*/)
tensor = torch.tensor([1,2,3])
tensor + 10
tensor - 1
tensor * 10
tensor / 2
torch.mul(tensor,10)
torch.add(tensor, 5)