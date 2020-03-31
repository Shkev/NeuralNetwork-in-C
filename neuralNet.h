#ifndef NEURALNET_H_
#define NEURALNET_H_

/*
 *An implementation of a simple neural network
 *in C to detect the pattern of the XOR function.
 *Stochastic Gradient Descent (SGD) is used as 
 *the optimizer.
*/

//activation function and its derivative
double sigmoid(double x);

double dSigmoid(double x);

//returns initialization values of all weights and biases between 0.0 and 1.0
double init_weight();

//initializing all weights and biases to between 0.0 and 1.0
void init_weights_bias(int numInputs, int numHiddenNodes, int numOutputs, double hiddenWeights[numInputs][numHiddenNodes], double outputWeights[numHiddenNodes][numOutputs], double* hiddenLayerBias, double* outputLayerBias);

//selection shuffle to randomize elements of array
void shuffle(int arr[], int size);

//returns root mean squared error (RMSE) for predictions in outputLayer.
double find_rmse(int numOutputs, int index, double training_outputs[][numOutputs], double* outputLayer);

//returns the average of all the values in the array
double avgArr(double arr[], int size);

//returns a prediction based on prediction inputs at index. Uses weigts and biases from trainined network.
double predict(int index, int numHiddenNodes, int numInputs, int numOutputs, double prediction_inputs[][numInputs], double hiddenWeights[][numHiddenNodes], double outputWeights[][numOutputs], double* hiddenLayerBias, double* outputLayerBias);

#endif
