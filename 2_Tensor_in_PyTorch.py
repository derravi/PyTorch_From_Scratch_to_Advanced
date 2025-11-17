"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#Tensors in PyTorch

# Whare are Tensors?
-> Tensor is a specializes multi_dimentional Array designed for mathematical and computational efficiancy.
-> Its a Generalization of the Array.
-> Dimention:- Koi Perticular tensor ketla direction ma spread out che tene.
-> Our Tensor spread in how many directions that is known as Dimentions.
    -> Ex.
    (i) 0D Tensor
        Scaler:- 0 Dimentional Array(Single Number):-
        -> Its Represent a Single number.
        -> Loss Value:- After a forwared pass loss functions compute a single scaler values, this is represent the differenvce between predicted and actual out put.
        -> 5,2.3,4,81
    (ii) 1D Tensor 
        Vecors:- 1 Dimentional Tensor:- This is a List of Numbers.
        -> The Array is the best example of the Vecores or one dimentional array.
        -> Its represent the sequance of collection of values.
        -> [1.2,-1.3,0.8]
    (iii) 2D Tensor
        Matrices:- 2 Dimentional Tensor
        -> Like one Grayscale image.
        -> [[1,2,3],
            [4,5,6]]
        -> This is a 2 Dimentional array OR 2 Dimentional Tensor
    (iv) 3D Tensor
        Colured Images:- 3D Array / 3D Tensor
        -> Ex. RGB Image:- Shape[125,255,3] 
        -> 125,255 is RGB Color and the 3 is dscribe a channels there is a 3D.

    (v) 4D Tensor
        Batches of RGB Images.
        -> Add the batch size as a additional dimentions into 3D data.
        ->(batch size x width X height x channels) 
        -> Batches of RGB Images:- [20,125,255,3]
        -> 20 is represent the how many RGB images we are use.
        -> 125,255 is represent the RGB Color.
        -> 3 its represent the channels.

    (vi) 5D Tensor 
        Video Data
        -> Adds a time dimentions  for data that chenges over time (eg. Video Frame)
        -> Video clip is represent a sequence of frames , where eatch frame is in RGB image.
        ->(video x frame x rgb x rgb x channel)
        -> Example:- [10,20,152,251,3]
        -> 10 is represent the Video
        -> 20 is represent the How many frame are there into the each video.
        -> 152,251 RGB color values.
        ->3 Represent the Chanels.

# Why are tensor UseFul?
    (i) Mathamatical Operations.
    -> Tensors is enable the mathamatical operations like add,mult,div,sub,dot_product,....ect

    (ii) Representations of Real world data.
    -> Its represent the Data like Audio,Video,Images,..etc into the Tensor.

    (iii) Efficiant Computatinos.
    -> Tensor are optimized for hardware acceleration, allowing computations on GPU or TPUs, which are crucial for training deep learning models.

#Where are Tensors use in Deep Learning?
    (i) Data Storage
    -> Training data is stored in Tensors. like images,video,audio.
    -> Text ko tensor me convert karke ham NLP me use kar sakte hain.
    -> Images ko tensor me convert karke ham image processing kar sakte hain.

    
    (ii) Weights and Biases
    -> The learnable parameter of a neural network (weight,biases) are stored in tensors.
    
    (iii) Matrix Operations
    -> Neural Network invoice operation like matrix multiplication, dot product and broadcasting -all  performed using tensors.
    
    (iv) Training Process
    -> During Forward passes , tensor flow through the network.
    -> Gradiens , represented as tensors are calculated during the backward pass.
#----------------------------------------------------------------------------------------------------------------------------------------------------------

# Instalation of the PyTorch into my Systems.
->  pip install torch `
"""
#To Check the Version of the Torch
# import torch
# print(torch.__version__)

#Check the GPU is Available or not.

# import torch

# if torch.cuda.is_available:
#     print("GPU is Available.")
#     print(f"GPU name is:{torch.cuda.get_device_name(0)}")
# else:
#     print("The GPU is not Available.")
#----------------------------------------------------------------------------------------------------------------------------------------------------------
"""
#How to Create a tensor into the PyTorch.

(1) Using empty.
-> we are use "empy()" function to make a empty tensor.

(2) Check type
-> This Function is use to check the Type of the veriable.

(3) Using zero
-> Make a tensor for the zeros values.

(4) Using ones
-> Make a tensor for the 1 values.

(5) Using rand
-> Use to make a tensor with the randome values of the 0 to 1.
-> Its give a different values into each run time.

(6) Using seed
-> Whenever we need a same values of the previus "rand()" Function so that time we are use this manual_seed() function.

(7) manual_seed

(8) Manual Tensor
-> we are able to make a custom tensor into the PyTorch.
-> we are use the "tensor()" function for making this custom tensor.

(9)arange() funciton
-> we are give the range for making a tensor.

(10) linspace()
-> Use to make a tensor like the lin space.

(11) eye
-> Its use to make a eye tensor using the "eye()" funciton.

(12) full
-> full() funciton is use to make a tensor for the perticular number.
-> like if we nedd to make a full matrics of the 5 with custom row,col.

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#Using Empty

import torch
#empty() Funciton.
# print(torch.empty(2,3))
# a  = torch.empty(2,3)

#type() Function.
# print(typef(a))

#zeros() functions.
# print(torch.zeros(3,2))

#ones() functions.
# print(torch.ones(3,2))

#rnad() functions.
# print(torch.rand(3,2))

#manual_seed() functions.
# torch.manual_seed(100)
# print(torch.rand(3,2))

# torch.manual_seed(100)
# print(torch.rand(3,2))

#tensor() function.
# print(torch.tensor([[1,2,3,4],[5,6,9,7]]))

#arange() Function
# print(f"Arange function:{torch.arange(0,10,1)}")

#linspace()
# print(f"Linspce tensor : {torch.linspace(0,10,10)}")

#eye() tensor
# print(f"eye tensor : {torch.eye(3,3)}")

#full() tensor
# print(f"Full tensor : {torch.full((5,5),2)}")
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------
"""
#tensor shape.
-> Use to find the shape of any tensor.
-> we are ue the "shape()" function to check the shape of the tensor.

#empty_like() :- its make the new tensor,its same tensor that we are giving into the parameter.
#zeros_like() :- 
#ones_like():-
#rand_line():- 

import torch

a = torch.zeros(3,3)
print("A values:",a)

print("Empty like A:",torch.empty_like(a))
print("Zero like A:",torch.zeros_like(a))
print("One Like A:",torch.ones_like(a))

#For rand_like() Function
x = a.to(torch.float64)
print("rand like A:",torch.rand_like(x))
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#Data Types

(1) type :- Check the Data types of the tensor.

(2) Type Changing into the Tensor.
-> Lets see one example of the Type Casting.
-> Ex. torch.tensor([1.0,2.3,5.7],dtype = torch.int32)
-> Its give me the int types of the data and also work as a int data types.

#Conver the data type od any pre define veriables.
-> we are use "to()" function to change the Data types of the Pre Define veriables.

import torch

# x = torch.tensor([1.2,2.5,3.4])
# print(x.dtype)

# x = torch.tensor([1.0,5.3,2.0,8.7],dtype=torch.int32)
# print("Data Type:",x.dtype)

# x = torch.tensor([1,2,4,3,5],dtype=torch.float64)
# print("Data types:",x.dtype)

# x = torch.tensor([1,2,3,4])
# y = x.to(torch.float64)
# print(y,y.dtype)

"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------
"""
#Mathamatical Operations in Tensor.

(1) Addition
(2) Subtractions
(3) Multiplication
(4) Division
(5) int division
(6) mod
(7) power

import torch

torch.manual_seed(100)
x = torch.rand(3,3)
print("Origonal Values:",x)

#addition
# print("Addition:",x+2)

#Subtraction
# print("Subtractoin:",x-1)

#Multiplication
# print("Multiplication:",x*5)

#Division
# print("Division:",x/5)


#int Division
# x = x*100
# print("int Division:",x/5)

#mod
# x = x*100
# print("Mod:",x%2)

#power
# x*=10
# print("Power:",x**2)
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------
"""
#Mathamatical Operations with 2 Tensor.

(1) Addition
(2) Subtractions
(3) Multiplication
(4) Division
(5) int division
(6) mod
(7) power

import torch

torch.manual_seed(100)
x = torch.rand(3,3)
y = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print("Origonal Values of X:",x)
# print("Origonal Values of y:",y)


#addition
# print("Addition:",x+y)

#Subtraction
# print("Subtractoin:",x-y)

#Multiplication
# print("Multiplication:",x*y)

#Division
# print("Division:",y/x)


#int Division
# x = x*100
# print("int Division:",y/x)

#mod
# x = x*100
# print("Mod:",x%y)

#power
# x*=10
# print("Power:",y**x)
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#Extra Functions. 

(1) abs() -> we are use the abs for convert the tensor into the absolute values.
-> Absolute it means make the same length of the all values into our tensor.

(2) negative() -> Its convert the negative value to positive and positive to negative.

(3) round() -> Its use to rouns our tensor values.

(4) ceil() -> Its round the values like if 5.78 = 6, 9.784 = 9.8

(5) floor() -> Reverse of the ceil functino.

(6) clamp() -> Its use for convert the data into some between range.
    -> G.S:- torch.clamp(tensor,min,max)

    
import torch

x = torch.tensor([12.35,85.15,-24.92,63.12,15.594])
# print("Absolute values:",torch.abs(x))

# print("Negative Funciton:",torch.negative(x))

# print("Round Values:",torch.round(x))

# print("Ceil data:",torch.ceil(x))

# print("floor data:",torch.floor(x))

# print("clamp function:",torch.clamp(x,min=10,max=20))
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------
"""
#Reduction Operations

(1) randint() :- its use to generte a randome tensor values between range of the given number.



import torch

torch.manual_seed(100)
# x = torch.randint(size=(3,3),low=1,high=10)
x = torch.randint(size=(3,3),low=1,high=10,dtype=torch.float32)

#sum() Function
#-> To sum full tensor.
# print("Sum of full tensor:",torch.sum(x))
# print("Sum of full tensor:",torch.sum(x,dim=0)) #dim=0 means colum wise sum.
# print("Sum of full tensor:",torch.sum(x,dim=1)) #dim=0 means row wise sum.

#mean() Function.
#-> use to find a mean for all and with row and column wise.

# print("Mean:",torch.mean(x))
# print("Mean:",torch.mean(x,dim=0)) #Col Wise Mean
# print("Mean:",torch.mean(x,dim=1)) #Row Wise Mean

#median() Function.

# # print("Median:",torch.median(x))
# print("Median:",torch.min(x))   #find the minimum values from the tensor.
# print("Median:",torch.max(x))   #find the maximum values from the tensor.

# #product & Standard Deviation.
# print("Product:",torch.prod(x))
# print("Standard Deviation:",torch.std(x))

#variance() Function.
# print("Variance:",torch.var(x))

#argmax and argmin
# print("Arg Max:",torch.argmax(x))
# print("Arg Min:",torch.argmin(x))

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#Matrix Operations.

import torch 

torch.manual_seed(100)
a = torch.randint(size=(3,3),low=1,high=20)

torch.manual_seed(101)
b = torch.randint(size=(3,3),low=1,high=20)

#Matrix Multiplications.
# print("Matrics Multiplications.",torch.matmul(a,b))

# #dot product
# c = torch.tensor([1,2,3])
# d = torch.tensor([4,5,6])
# print("Dot Product:",torch.dot(c,d))

#transpose:-
# print("Transpose:",torch.transpose(a,0,1))

#determenent
# print("Determenant:",torch.det(a))

#invers
# print("Invers:",torch.inverse(a))

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#(5) Comparision Operations.

import torch 

torch.manual_seed(100)
a = torch.randint(size=(3,3),low=1,high=20)

torch.manual_seed(101)
b = torch.randint(size=(3,3),low=1,high=20)

#grater than
print("grater than",a>b)
#less than
print("Less Than",a<b)
#equla to
print("Equla to",a==b)
#not equal to
print("Not equal to",a!=b)
#grater then equal to
print("Grater then equal to",a>=b)
#less than equal to
print("Less than equal to",a<=b)

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#Special Functions
import torch

torch.manual_seed(100)
k = torch.randint(size=(3,3,),low=1,high=10,dtype=torch.float64)

#log
# print("Log",torch.log(k))

#exponents exp()
# print("Exponent:",torch.exp(k))

#sqrt - Square Root
# print("Square root:",torch.sqrt(k))

#sigmoid()
# print("Sigmoid:",torch.sigmoid(k))

#softmax()
# print("Softmax:",torch.softmax(k,dim=0)) #0 for rows
# print("Softmax:",torch.softmax(k,dim=1)) #1 for col

#relu()
# print("Relu:",torch.relu(k))

# clone() Function.
#-> This is use to make a copy of the other tensor.

print(f"ID of k is {id(k)} and the value of k is {k}")

b = k.clone()
b[0][0] = 50
print(f"ID of b is {id(b)} and the value of b is {b}")
"""

"""
#Check the GPU is Available or not into the Systems.
import torch

if torch.cuda.is_available:
    print("Yes")
else:
    print("No")


#Perform a examples on the GPU
import torch

device = torch.device('cuda')

torch.manual_seed(100)
a = torch.rand((2,3),device=device)
print(a)


#Lest see one example of the CPU and GPU performance with the Timeline.

import torch
import time

size = 1000

torch.manual_seed(100)
cpu1 = torch.randint(size=(size,size),low=1,high=5)
torch.manual_seed(101)
cpu2 = torch.randint(size=(size,size),low=1,high=5)

#Masure the time.
start_time = time.time()
result_cpu = torch.matmul(cpu1,cpu2)
cpu_time = time.time() - start_time

print(f"Time taken by CPU is:{cpu_time}")

gpu1 = cpu1.float().to('cuda')
gpu2 = cpu2.float().to('cuda')

#Masure the time.
start_time = time.time()
result_gpu = torch.matmul(gpu1,gpu2)
gpu_time = time.time() - start_time

print(f"Time taken by GPU is:{gpu_time}")
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------

#Reshape the Tensors

import torch

a = torch.full((4,4),5)

#reshape
# print("Reshape Tensor:",a.reshape(2,2,2,2))

#Flatter :- Convert the multidimentional tensor into the one dimentional tensor.
# print("Flatten Tensor:",a.flatten())

#permute :- Its use to reshape the rows and columns.
# b = torch.rand(2,3,5)
# print(b.permute(2,0,1).shape)

#unsqueeze
# Like Images size

# b = torch.rand(226,226,3)
# print(b.unsqueeze(0).shape)

#squeeze
# c = torch.rand(1,40)
# print(c.squeeze(0).shape)

#How to convert the Tensor to Numbpy and vise a versa.

import numpy as np
import torch
"""
#To convert the Torch tensor into the Numpy array.
a = torch.tensor([1,2,3,4])
print(f"The Torch Tensor is {a}")
print(f"The Type of Tensor is {type(a)}")
b = a.numpy()
print(f"The Numpy array is : {b}")
print(f"The Type of array is {type(b)}")
"""

#Convert the Numpy array to torch
#-> from_numpy() This function is use to convert the Numpy arrys to Tensor.

a = np.array([1,2,3,4,5])
print(f"The Numpy array is : {a}")
print(f"The Type of array is {type(a)}")

b = torch.from_numpy(a)
print(f"The Torch Tensor is {b}")
print(f"The Type of Tensor is {type(b)}")