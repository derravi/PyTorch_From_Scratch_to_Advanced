import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("breast-cancer.csv")
print(df.head())

print(f"The total number of Columns is {df.shape[1]} and the total number of Rows is {df.shape[0]}.")

df.drop(columns=['id'],inplace=True)

print(f"The total number of Columns is {df.shape[1]} and the total number of Rows is {df.shape[0]}.")

df['diagnosis'].unique()

x = df.iloc[:,1:] #All Colum Excepted 'diagnosis'
y = df.iloc[:,0] # 'diagnosis' Column

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

#Conver the all numpy Arrays to the Tensors.
x_train_tensor = torch.from_numpy(x_train)
x_test_tensor = torch.from_numpy(x_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

print(x_test_tensor.shape)

class MyfirstNN():
    def __init__(self,x):
        self.weights = torch.rand(x.shape[1],dtype=torch.float64,requires_grad=True)
        self.bias = torch.zeros(1,dtype=torch.float64,requires_grad=True)

    def forward(self,x):
        z = torch.matmul(x,self.weights) + self.bias
        y_predict = torch.sigmoid(z)
        return y_predict

    def loss_function(self,y_pred,y):
        #Clamp Predictions to avoid log(0)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred,epsilon,1-epsilon)

        #calculate loss
        loss = -(y_train_tensor*torch.log(y_pred) + (1 - y_train_tensor) * torch.log(1 - y_pred)).mean()
        return loss

learning_rate = 0.1
epochs = 20

#Create Model
model = MyfirstNN(x_train_tensor)

#Define Loop
for i in range(epochs):
 
    #Forward Pass
    y_predict = model.forward(x_train_tensor)
 
    #Loss Calculation
    loss = model.loss_function(y_predict,y_train_tensor)
 
    #Backward Pass
    loss.backward()
 
    #Parameters Updates.
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad

    #zero gradients
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    #Print Loss in each epoch
    print(f"Epoch: {i + 1}, Loss : {loss.item()}.")

print(f"Model Weights:{model.weights}")

print(f"Model Bias:{model.bias}")

#Evaluation of the Model

with torch.no_grad():
    y_pred = model.forward(x_test_tensor)
    y_pred = (y_pred>0.9).float()

    #Accuracy score of the ML NN Pipeline
    accuracy = (y_pred == y_test_tensor).float().mean()
    print(f"The Accuracy is : {accuracy}")
