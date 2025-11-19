#PyTorch Autograd

"""
# Why we need a PyTorch Autograd? :- Auto Matic machine for Automatic Gradient Calculation.
->This is a Core concept of the PyTorch its provides automatic differentiation for tensor operation.
-> Its enable gradiant computation , which is essential for training machine learning models using optimization algoridhms like gradiant descrate.

->Autograd basically एक automatic differentiation engine है. इसका काम है हर operation का gradient (यानि derivatives) निकालना. जब भी आप कोई forward pass करते हो, autograd पीछे-पीछे सब gradient की bookkeeping करता जाता है.

#Lets see how to find the Values of the Any derivatives using the PyTorch.
-> First we are make one Tensor fot do this, and set the "requires_grad" is "True" this is notify that its find the derivatives.

# Forward Propagation (Forward Pass)
-> It is the process where input data passes through the neural network layer by layer to produce an output.

# Backward Propagation (Backward Pass / Backpropagation)
-> It is the process of calculating gradients based on the loss and updating the model’s weights to reduce the error.
->Backpropagation असल में एक algorithm है, जो इन gradients को use करके neural network के weights और biases को update करता है. यानि autograd gradients निकालता है, और backpropagation उन gradients का इस्तेमाल करके network को सिखाता है कि अगली बार prediction बेहतर कैसे करनी है.

## Derivative = output kitna change hota hai jab input/weight badalta hai
-> derivatives का सीधा-साधा मतलब है कि जब आपका neural network कुछ output देता है, तो ये पता लगाना कि अगर आप उसके input या weights को थोड़ा सा बदलें, तो output कितना बदलेगा. यानी derivatives उस बदलाव की दर दिखाते हैं.
-> Neural networks में derivatives इसीलिए ज़रूरी हैं क्योंकि इनकी मदद से model ये सीखता है कि उसे अपने weights को कैसे adjust करना है ताकि उसका output सही दिशा में जाए. मतलब जब आप loss को कम करना चाहते हैं, तो derivatives बताते हैं कि किस direction में weights को बदलना है.

## Gradient = ACtual value of the Derivative.
-> 
## Auto-Grad = PyTorch ka system jo ye gradients khud calculate karta hai


#Example - 1 (Scaler Input)

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


#Example - 2 (Scaler Input)

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

#Example - 3 (Scaler Input)
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

#Print the Grade for the W and B. Gradiantmeans the Actual Value of the Derivative.
print("\nThe Value of W grade is :",w.grad)
print("The Value of b gr2ade is :",b.grad)


#Vector Input Tensors.
import torch

x = torch.tensor([1.0,2.0,3.0],requires_grad=True)

y = (x**2).mean()

y.backward()

print("Value of X",x)
print("Value of Y",y)
#Lets Check manualy
#  (1**2)/3 + (2**2)/3 + (3**2)/3 
# (0.33+1.33+3) y = 4.66 (Forword Direction.)
print("Value of Gradiant:",x.grad)

#backword directions.
#Derivative of x**2 = 2x/3
# (1*2)/3 = 0.6666
# (2*2)/3 = 1.3333
# (3*2)/3 = 2.0

#Whenever we are run backword and print gradiant so its make a issues , the issues is that the gradiant is added into the previus gradiant.
#-> Like we are find the gradiant of 3 is 9 but we are re-run the same code so its print the gradiant is 18. its add the previus one also.
#-> So resolve this problem we are use "zeros_()" function.
#-> like the belov code.

import torch

x = torch.tensor(3.0,requires_grad=True)
y = x**2
y.backward()
print(x.grad)
x.grad.zero_()
"""

#How to disable Gradiant Tracking
#-> Whenever we dont need to find the gradiant that time we are use this.
#-> Whenever we are predict the Tensors that time we dont need to track the gradiant so that time we are stop the gradiant tracking.
#-> Lets asume one senario , you are completely find one gradiant and then we are re use the backword tracking that time we dont need to find the gradiant.

import torch

x = torch.tensor(3.0,requires_grad=True)
y = x**2
print(x)
print(y)
y.backward()
print(x.grad)

Oprion 1 :- requires_grad=False
x.requires_grad_(False)
print(x)
y = x**3
y.backward()

Option 2 :- detach()
x = torch.tensor(5.0,requires_grad=True)
z = x.detach()
y = x**2

print(y)

y1 = z**2

y.backward()
print(x.grad)

y1.backword()
print(z.grad)

Option 3 :- torch.no_grad()

x = torch.tensor(6.0,requires_grad=True)

with torch.no_grad():
    y = x**2
print(y)
y.backward()
print(x.grad)