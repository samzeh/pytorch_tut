# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3)  Training loop
#   - forward pass: compute prediction
#   - backward pass: compute gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 1) model

# creating the model with sckit-learn dataset
# n_samples = number of data points
# n_features = number of input features for each sample (one variable) ex) y=mx+b -> 1 feature (x)
# noise = add randomness to the data points
# random_state = If omitted, dataset changes every run

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0],1) # reshape y to be a column vector (.view changes shape without changing data)

n_samples, n_features = X.shape

input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

# 2) loss and optimizer

learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 100

for epoch in range(num_epochs):
  #forward pass and loss
  y_predicted = model(X) # technically runs model.forward(X)
  loss = criterion(y_predicted, y)

  #backward pass
  loss.backward()

  #update
  optimizer.step() # updates weight
  optimizer.zero_grad()

  if (epoch + 1) % 10 == 0:
    print(f'epoch {epoch+1}, loss = {loss.item():.4f}')

# plot
predicted = model(X).detach().numpy() # detach stops tracking for gradients!
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()