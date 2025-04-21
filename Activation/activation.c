#include "activation.h"
#include <math.h>

double SigmoidFunc(double x) { return 1 / (1 + exp(-x)); }
