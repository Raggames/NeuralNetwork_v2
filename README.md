# Neural Network Project

A small neural network/machine learning self-educational project (work-in-progress) based on the Microsoft back propagation demo (2013). 
(The original code is under NeuralNetwork/Assets/Script/BuildNeuralNetworkDemo.cs)
I do not own this code, I just began by analysing and rewriting all the code in my own way to understand it, and then slowly extends its functionnalities.

The project is made under unity with the goal of understanding machine learning technology.

The neural network can be trained either by backpropagation (with weight decay, momentum, and a learning rate decay feature to avoid overlearning) or by a simple genetic algorithm.
The network architecture can be saved as a Scriptable Object.
The network is Fully-Connected and Multilayer.
 
Further improvments are intended : 
* <s>adding a multi-layer fully connected implementation </s>
* <s>abstracting the training setup as Scriptable Object (accuracy function, heuristics...)</s>

* a more complex genetic trainer (mutation, crossing DNA,randomization of training parameters, adding 'momentum' from parent-to-child weight updates)..
* optimizing computations
* testing image denoizing 
* some custom interface work
* creating a Network Runner (to run tests on a trained network easily)
* Samples for Odd of Even number detection, flower classification and a simple function regression can be used to test the learning process using backpropagation or genetic algorithm.
