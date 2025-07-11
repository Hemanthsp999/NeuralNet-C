/* activation.h */
#ifndef Activation_
#define Activation_
#include "../Neural/neural.h"

float sigmoid(float);
float sigmoid_derivative(float);
float tan_h(float);
float tanh_derivative(float);
float relu(float);
float categorical_cross_entropy(Layer *, Layer *);
float binary_cross_entropy(Layer *, Layer *);
float mse(Layer *, float *);
void soft_max(Layer *);
void l2_regularization(NeuralNetwork *, float);
void _init_adam_optimizer(NeuralNetwork *);
void __adam_update(NeuralNetwork *, float, int);
float get_weight_gradient(NeuralNetwork *, size_t, size_t, size_t);

#endif
