#include <stdio.h>
#include <stdlib.h>
#include "neuralNet.h"

/*HYPERPARAMS*/
static const int epochs = 10000;
static const double learning_rate = 0.1;

static const int numInputs = 2;
static const int numHiddenNodes = 2;
static const int numOutputs = 1;
static const int trainingSets = 4;

static const int numPredictInputs = 2;

double hiddenLayer[numHiddenNodes];
double outputLayer[numOutputs];

double hiddenLayerBias[numHiddenNodes];
double outputLayerBias[numOutputs];

double hiddenWeights[numInputs][numHiddenNodes];
double outputWeights[numHiddenNodes][numOutputs];

//training data
double training_inputs[trainingSets][numInputs] = { {0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f} };
double training_outputs[trainingSets][numOutputs] = { {0.0f},{1.0f},{1.0f},{0.0f} };

//prediction data
double prediction_inputs[][numPredictInputs] = { {0, 1} };

int main()
{
  int loopCount, x, j, k;
  double rmse[numInputs];
  init_weights_bias(numInputs, numHiddenNodes, numOutputs, hiddenWeights, outputWeights, hiddenLayerBias, outputLayerBias);
  
  //iterate through all training data for a number of epochs
  for(loopCount = 0; loopCount < epochs; loopCount++)
  {
    //shuffle training data for SGD
    int trainingSetOrder[trainingSets];
    //creating elements of trainingSetOrder array
    for(x = 0; x < trainingSets; x++)
    {
      trainingSetOrder[x] = x;
    }

    shuffle(trainingSetOrder, trainingSets);

    //cycling through each training set element
    for(x = 0; x < trainingSets; x++)
    {
      int index = trainingSetOrder[x];

      //computing hidden layer activation
      for(j = 0; j < numHiddenNodes; j++)
      {
	double activation = hiddenLayerBias[j];
	for(k = 0; k < numInputs; k++)
	{
	  activation += training_inputs[index][k] * hiddenWeights[k][j];
	}
	hiddenLayer[j] = sigmoid(activation);
      }

      //compute output layer activation
      for(j = 0; j < numOutputs; j++)
      {
	double activation = hiddenLayerBias[j];
	for(k = 0; k < numHiddenNodes; k++)
	{
	  activation += hiddenLayer[k] * outputWeights[k][j];
	}
	outputLayer[j] = sigmoid(activation);
      }
      
      //Compute change in output weights
      double deltaOutput[numOutputs];
      for(j = 0; j < numOutputs; j++)
      {
	//the derivative of the error; using Mean Square Error (MSE)
	double dError = (training_outputs[index][j] - outputLayer[j]);
	//calculating the RMSE
	deltaOutput[j] = dError * dSigmoid(outputLayer[j]);
      }
      //Compute change in hidden layer weights
      double deltaHidden[numHiddenNodes];
      for(j = 0; j < numHiddenNodes; j++)
      {
	double dError = 0.0f;
	for(k = 0; k < numOutputs; k++)
	{
	  dError += deltaOutput[k] * outputWeights[j][k];
	}
	deltaHidden[j] = dError * dSigmoid(hiddenLayer[j]);
      }

      //apply change in output weights to improve predictions
      for(j = 0; j < numOutputs; j++)
      {
	outputLayerBias[j] += deltaOutput[j] * learning_rate;
	for(k = 0; k < numHiddenNodes; k++)
	{
	  outputWeights[k][j] = hiddenLayer[k] * deltaOutput[j] * learning_rate;
	}
      }

      //apply change in hidden weights
      for(j = 0; j < numHiddenNodes; j++)
      {
	hiddenLayerBias[j] = deltaHidden[j] * learning_rate;
	for(k = 0; k < numInputs; k++)
	{
	  hiddenWeights[k][j] += training_inputs[index][k] * deltaHidden[j] * learning_rate;
	}
      }
      printf("%f\n", outputLayer[0]);
      rmse[index] = find_rmse(numOutputs, index, training_outputs, outputLayer);
    }
    //printf("%d / %d :: %f\n", loopCount+1, epochs, avgArr(rmse, numInputs));
  }
  printf("Final Loss :: %f\n", avgArr(rmse, numInputs));

  //Making a prediction based on data
  double prediction = predict(0, numHiddenNodes, numPredictInputs, numOutputs, prediction_inputs, hiddenWeights, outputWeights, hiddenLayerBias, outputLayerBias);
  printf("Prediction for {%f, %f}: %f\n", prediction_inputs[0][0], prediction_inputs[0][1], prediction);
  exit(EXIT_SUCCESS);
}
