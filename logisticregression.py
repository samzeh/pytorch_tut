# logistic regression is used for binary classification
# e.g. (0 or 1, yes or no, true or false)
# predicts a probability between 0 and 1
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data

bc = datasets.load_breast_cancer()
print(bc.keys()) # View all the things available in dataset.
# Ex) print(bc.DESCR) prints the description of the dataset

X, y = bc.data, bc.target

n_samples, n_features = X.shape # sample is the specific tumour, feature is the attribute/characteristic of the tumour

print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale

sc = StandardScaler() # scales data so that each feature has mean 0 and variance 1. sc is an object of the class StandardScaler
X_train = sc.fit_transform(X_train) #fits data (calculates mean and std) then transforms/normalizes all the data in training set
X_test = sc.transform(X_test) # transforming/normalizing test set based on training set mean and std
#^^^ we ONLY do transform on test to avoid data leakaage

X_train = torch.from_numpy(X_train.astype(np.float32)) # must use type float32 for pytorch
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1) # rehaping to pytorch with n_samples, 1 feature (column)
y_test = y_test.view(y_test.shape[0],1)

# 1) model
# f = wx + b, sigmoid at the end

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x)) # turns the values into something between 0 or 1
        return y_predicted

model = LogisticRegression(n_features)

# 2) loss and optimizer

learning_rate = 0.01
criterion = nn.BCELoss() # binary cross entropy loss (for binary classification)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 100

# 3) training loop

for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train) # technically model.forward(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    
    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
   y_predicted = model(X_test)
   y_predicted_cls = y_predicted.round() # rounds to nearest integer
   acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
   print(f'accuracy = {acc:.4f}')



