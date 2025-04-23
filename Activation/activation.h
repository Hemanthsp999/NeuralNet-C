#ifndef Activation_
#define Activation_

const int zero = 0;
#define max(zero, x) ((x > zero) ? x : zero)

double SigmoidFunc(double x);
double tan_h(double x);
double relu(double);

#endif
