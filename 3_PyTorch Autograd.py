#PyTorch Autograd

"""
# Why we need a PyTorch Autograd?
->This is a Core concept of the PyTorch its provides automatic differentiation for tensor operation.
-> Its enable gradiant computation , which is essential for training machine learning models using optimization algoridhms like gradiant descrate.

#Lets see how to find the Values of the Any derivatives using the PyTorch.
-> First we are make one Tensor fot do this, and set the "requires_grad" is "True" this is notify that its find the derivatives.

# Forward Propagation (Forward Pass)
-> It is the process where input data passes through the neural network layer by layer to produce an output.

# Backward Propagation (Backward Pass / Backpropagation)
-> It is the process of calculating gradients based on the loss and updating the model’s weights to reduce the error.

#Example - 1

import torch

# x = torch.tensor(3.0,requires_grad = True)
x = torch.tensor([1,2,3,4],dtype=torch.float64,requires_grad=True)
y = x**2

print(f"Value of x is {x}")

#Forword Direction
print(f"Value of y is {y}") #Forword

#Backword Direction :- Its use to find the derivative of the Given "X" Equestion.
#We are use "backwor()" Function to convert the Equestion intot the darivative.

y.backward() #make the derivative of the "Y" Equestion.

# We are use "grade()" Function to check the value of the Derivatives for the speccific values.
print(f"Value of the Derivative base ont the value {x} is {x.grad}.")


#Example - 2

import torch

x = torch.tensor(3.0,requires_grad=True)

y = x**2

z = torch.sin(y)

print("Forword y = ",y)
print("Forword z = ",z)

#Backword.
z.backward()
print(x.grad)
"""

#Example - 3
"""

(x)--------->(w)--------->(Sigmoid - (b) )--------->y_predict------------->(loss)

(1) Linear Transformation:-
    z = ( X * W ) + b

(2) Activation (Sigmoid Function)
    y_predict = 6(z) = a / (a + e**(-z))

(3) Loss Function (Binory Cross Entropy-Loss)
    L = - [y_target * ln(y_predict) + (1 - y_target) * ln(1 - y_predict)]

#Computation Graph

------>------------>-------->---------------->-------->Forword Direction------------->----------------->

(b)-----------------
                   |
(x)------(*)------(+)-------(z)-------(Sigmoid)---------(Y_predict)--------(Loss_function)--------(Loss)
          |                                                                     |
          |                                                                     |
         (w)                                                                   (y) (Actual)
------<----------<-----------<-----------<------------->BackWord Direction------------<-----------------<

#Use The Given Equestions of the Coding.

-> User Veriable with  the Actual And Predictable data.
x- tensor
y- tensor
w- tensor(requeres_grade=True)
b- tensor(requeres_grade=True)

(1) z = (x*w) + b
(2) y_predict = sigmoid(z)
(3) loss = binory_corss_entropy_loss(y_predict,t-Actual)

"""

import torch
import torch.nn.functional as F

X = torch.tensor(6.9)
Y = torch.tensor(0.0)

w = torch.tensor(1.0,requires_grad=True)
b = torch.tensor(0.0,requires_grad=True)

print("\nValue of the X :",X)
print("Value of the Y :",Y)

print("\nValue of the w :",w)
print("Value of the b :",b)

#We are use the Z Equestion.
z = (X*w) + b
print("\nValue of the Z :",z)

#we are use the Y_Predict Equestion.
Y_predict = torch.sigmoid(z)
print("\nThe value of the Y_predict is:",Y_predict)

#Find the Loss using the Given Equestions.
#-> We are us the "binory_cross_entropy_loss()" Function to find the Loss.
# We are put a parameter that is the "Predicted y" and actual value of "y".
 
loss = F.binary_cross_entropy(Y_predict,Y)
print("\nThe Value of the Loss is :",loss)

#Do the Backword
loss.backward()

#Print the Grade for the W and B.
print("\nThe Value of W grade is :",w.grad)
print("The Value of b grade is :",b.grad)