#include <stdio.h>
#include <stdlib.h>
#include "neuralNet.h"

double sigmoid(double x)
{
  return 1/(1+exp(-x));
}

double dSigmoid(double x)
{
  return x * (1 - x);
}

double init_weight()
{
  return (((double)rand()) / ((double)RAND_MAX))
}
