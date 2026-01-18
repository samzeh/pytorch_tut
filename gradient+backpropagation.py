import torch

x=torch.randn(3, requires_grad=True) # 1D tensor with 3 rand values & enable gradient tracking
print(x) # Create a 1D tensor with 3 random values from a normal distribution

y=x+2
print(y) # y is also a tensor with grad tracking

z=y*y*2
print(z) # z is also a tensor with grad tracking

z=z.mean()
print(z) # z is also a tensor with grad tracking

z.backward() # Calculates dz/dx so z must be a scalar!
print(x.grad) # Actually prints the gradients of x


x.grad.zero_() # Reset gradients to zero MUST DO THIS AT THE END

#--------------------Disabling Gradient Tracking--------------------

x.requires_grad_(False) #whenever func has trailing underscore, it means it will modify variable in place
y=x.detach() # creates a new tensor that does not require gradient
with torch.no_grad(): #temporarily set all the requires_grad to false
    y=x+2
    print(y)


#--------------------Back Propagation--------------------

# Problem backpropagation solves: 

# Which parameters should I change, and by how much, to reduce the mistake give 
# some numbers and a model that uses them?

# LOSS -> PARAMETERS (backwards) rather than PARAMETERS -> LOSS (forwards)

#-----------------Linear Regressino from Scratch-----------------
import numpy as np

# f = w*x
# f = 2*x

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w=0.0

# model prediction

def forward(x):
    return w*x

# loss = Mean Squared Error (MSE)

def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

# gradient
# MSE = 1/N * (wx-y)^2
# dJ/dw = 1/N * 2x(wx-y) where J is the MSE function

def gradient(x, y, y_predicted): # manual backpropagation
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training

learning_rate = 0.01
n_iters = 20 #number of iterations

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred) # positive dw = increase w increase loss

    # update weights
    w -= learning_rate * dw 

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}') 
    
print(f'Prediction after training: f(5) = {forward(5):.3f}')

#-----------------Linear Regression with PyTorch-----------------

import torch

# f = w*x
# f = 2*x

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)

# model prediction

def forward(x):
    return w*x

# loss = Mean Squared Error (MSE)

def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training

learning_rate = 0.01
n_iters = 100 #number of iterations

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    # gradients
    l.backward() # dl/dw

    # update weights
    with torch.no_grad(): 
        w -= learning_rate * w.grad 
    
    w.grad.zero_() # Reset gradients to zero MUST DO THIS AT THE END
    
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}') 
    
print(f'Prediction after training: f(5) = {forward(5):.3f}')

#-----------------Linear Regression using nn.Module-----------------

# 1) Design model (input, output siz,e forward pass)
# 2) Construct loss and optimizer
# 3)  Training loop
#   - forward pass: compute prediction
#   - backward pass: compute gradients
#   - update weights

import torch
import torch.nn as nn # neural network module

# f = w*x
# f = 2*x

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype = torch.float32)

# features = column, samples = rows
# X.shape -> (4, 1) 4 samples, 1 feature OR 4 rows, 1 column

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size) DOES SAME THING AS BELOW

class LinearRegression(nn.Module): # create custom model
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        #define layers
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training

learning_rate = 0.01
n_iters = 100 #number of iterations

loss = nn.MSELoss() # creates an object of class nn.MSELoss to use to calculate loss later
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # stochastic gradient descent

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    # gradients
    l.backward() # dl/dw

    # update weights
    optimizer.step()
    optimizer.zero_grad() # Reset gradients to zero MUST DO THIS AT THE END
    
    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}') 
    
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')