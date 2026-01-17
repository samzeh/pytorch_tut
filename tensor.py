import torch
import numpy as np

#tensor = multidimentional array

#--------------------Creating Tensors--------------------

x = torch.empty(1) # Create an empty tensor with a scalar value
x=torch.rand(2,2) # Create a 2x2 tensor with random values
x=torch.zeros(2,2) # Create a 2x2 tensor with zeros
x=torch.ones(2,2, dtype=torch.int) # Create a 2x2 tensor with ones
y=torch.tensor([2.5, 0.1])

print(x.dtype) # Print the data type of the tensor
print(x.size()) # Print the size of the tensor
print(y)

#--------------------Tensor Operations--------------------
x=torch.rand(2,2)
y=torch.rand(2,2)
print(x)
print(y)
y.add_(x) # Inplace addition (modifies y and adds all elements of x to y)
z=x+y # or torch.add(x,y)

z = x-y # or torch.sub(x,y)
z = x*y # or torch.mul(x,y)
z = x/y # or torch.div(x,y)

print(z)

x=torch.rand(5,3)
print(x)
print(x[1,:]) # Print row number INDEX 1 but all columns
print(x[1,1]) # Print element at row 1 and column 1
print(x[1,1].item()) # Get the actual value

#--------------------Reshaping Tensors--------------------

x=torch.rand(4,4)
print(x)
y=x.view(16)
print(y)

a = torch.ones(5)
print(a)
b=a.numpy()
print(type(b)) # Convert tensor to numpy array

a=np.ones(5)
print(a)
b=torch.from_numpy(a)
print(b) # Convert numpy array to tensor

