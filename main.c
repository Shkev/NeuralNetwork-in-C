#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "neuralNet.h"

static const int numInputs = 2;
static const int numHiddenNodes = 2;
static const int numOutputs = 1;
static const int epochs = 10000;
static const int trainingSets = 4;

double hiddenLayer[numHiddenNodes];
double outputLayer[numOutputs];

double hiddenLayerBias[numHiddenNodes];
double outputLayerBias[numOutputs];

double hiddenWeights[numInputs][numHiddenNodes];
double outputWeights[numHiddenNodes][numOutputs];

//training data
double training_inputs[numTrainingSets][numInputs] = { {0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f} };
double training_outputs[numTrainingSets][numOutputs] = { {0.0f},{1.0f},{1.0f},{0.0f} };

int main()
{
  //iterate through all training data for a number of epochs
  for(int loopCount = 0; loopCount < epochs; loopCount++)
  {
    //shuffle training data for SGD
    
  }
  exit(EXIT_SUCCESS);
}
