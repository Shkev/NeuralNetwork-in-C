#ifndef NEURALNET_H_
#define NEURALNET_H_

/*
 *An implementation of a simple neural network
 *in C to detect the pattern of the XOR function.
 *Stochastic Gradient Descent (SGD) is used as 
 *the optimizer.*/

//activation function and its derivative
double sigmoid(double x);

double dSigmoid(double x);

//Initializing all weights and biases between 0.0 and 1.0
double init_weight();




#endif
