"""
# NN Module :- Nural Network Module
# torch.optim module
-> An NN module is a small part of a neural network that takes some input, does a calculation, and gives an output. Many modules are combined together to build a full neural network.
-> NN Module = Neural Network ka part (layer/function) jo input leta hai, kuch operation karta hai, aur output deta hai.
Many modules → full neural network.
-> In PyTorch, an NN module is a building block used to create neural networks. Every layer (like Linear, Conv2d, ReLU, etc.) is written as a module. You combine many modules to build a full model.

->Ex.
nn.Linear → linear layer module
nn.ReLU → activation module
nn.Conv2d → convolution module
nn.Dropout → dropout module
nn.Sequential → modules ko ek line me jodna
nn.Module → base class jisse sare modules bante hain

#Lets see the Key Component of this NN Module.

(1) Modules(Layers):-
-> NN module = one piece of a neural network that performs a specific task, and you connect multiple pieces to create the complete model.
-> Common Layers :- Whenever we are use some layers into multiple times into our Nural Network so this is called Common Layers that we are use a multiple times into our NN or you cany say that we are use those layers into the each nural network. So this is our common layers.
    -> like:- nn.Linear,nn.Conv2d,nn.LSTM,

(2) Activation Function:-
-> Neural network me activation function ek aisa function hota hai jo decide karta hai ki ek neuron ka output kya hoga. Yeh input me kuch non-linearity add karta hai, taaki neural network complex patterns seekh sake.
-> Socho neural network ke neurons ko switches ki tarah, aur activation function ko ek aise controller ki tarah jo decide karta hai ki switch kab on hoga aur kab off. Yeh ensure karta hai ki network linear nahi, balki complex patterns bhi samajh paye.

(3) Loss Function:-
-> Its Provide loss function such as nn.CrossEntropyLoss,nn.MSELoss and nn.NLLLoss to quantify the difference between the models predictions and actual targets.

(4) Container Modules:- When we are use multiple layers for making a NN so we are able to make one container of this all layers.
-> nn.Sequential :- A sequancial container it takes layers in order.

(5) Regularization and dropout.
-> Layers like nn.Dropout and nn.BatchNorm2d help prevent overfitting and improv the models ability to generalize to new data.

# torch.optim
-> torch.optim is a collection of optimizers in PyTorch that adjust the model’s weights during training so the model learns better.


"""