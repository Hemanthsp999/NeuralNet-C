#include "activation.h"
#include <math.h>

double SigmoidFunc(double x) { return 1 / (1 + exp(-x)); }

double tan_h(double x) { return (sin(x) / cos(x)); }

double relu(double x) { return max(0, x); }
