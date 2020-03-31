#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "neuralNet.h"

double sigmoid(double x)
{
  return 1/(1 + exp(-x));
}

double dSigmoid(double x)
{
  return x * (1 - x);
}

double init_weight_val()
{
  srand(time(0));
  return (double)rand() / (double)RAND_MAX;
}

void init_weights_bias(int numInputs, int numHiddenNodes, int numOutputs, double hiddenWeights[numInputs][numHiddenNodes], double outputWeights[numHiddenNodes][numOutputs], double* hiddenLayerBias, double* outputLayerBias)
{
  //initializing weights
  int j, k;
  for(j = 0; j < numInputs; j++)
  {
    for(k = 0; k < numHiddenNodes; k++)
    {
      hiddenWeights[j][k] = init_weight_val();
    }
  }
  for(j = 0; j < numHiddenNodes; j++)
  {
    for(k = 0; k < numOutputs; k++)
    {
      outputWeights[j][k] = init_weight_val();
    }
  }

  //initializing biases
  for(j = 0; j < numHiddenNodes; j++)
  {
    hiddenLayerBias[j] = init_weight_val();
  }
  for(j = 0; j < numOutputs; j++)
  {
    outputLayerBias[j] = init_weight_val();
  }
}

void shuffle(int arr[], int size)
{
  //to randomize seed to prevent getting the same results every time
  srand(time(0));
  int k;
  for(k = size - 1; k > 0; k--)
  {
    int r = rand() % (k+1);
    assert(r <= k);

    int tmp = arr[k];
    arr[k] = arr[r];
    arr[r] = tmp;
  }
}

double find_rmse(int numOutputs, int index, double training_outputs[][numOutputs], double* outputLayer)
{
  double avgDiff = 0.0f;
  int counter;
  for(counter = 0; counter < numOutputs; counter++)
  {
    //if there is only one output per batch
    avgDiff += pow(outputLayer[counter] - training_outputs[index][counter], 2);
  }
  avgDiff /= numOutputs;

  return sqrt(avgDiff);
}

double avgArr(double arr[], int size)
{
  int counter;
  double avg = 0.0f;
  for(counter = 0; counter < size; counter++)
  {
    avg += arr[counter];
  }
  return avg/size;
}

double predict(int index, int numHiddenNodes, int numInputs, int numOutputs, double prediction_inputs[][numInputs], double hiddenWeights[][numHiddenNodes], double outputWeights[][numOutputs], double* hiddenLayerBias, double* outputLayerBias)
{
  int k, j;
  double prediction, hiddenLayer[numHiddenNodes];
  //running prediction input through hidden layer
  for(j = 0; j < numHiddenNodes; j++)
  {
    prediction = hiddenLayerBias[j];
    for(k = 0; k < numInputs; k++)
    {
      prediction += prediction_inputs[index][k] * hiddenWeights[k][j];
    }
    hiddenLayer[j] = sigmoid(prediction);
  }

  //computing outputs in output layer
  for(j = 0; j < numOutputs; j++)
  {
    prediction = hiddenLayerBias[j];
    for(k = 0; k < numHiddenNodes; k++)
    {
      prediction += hiddenLayer[k] * outputWeights[k][j];
    }
    prediction = sigmoid(prediction);
  }

  return prediction;
}
