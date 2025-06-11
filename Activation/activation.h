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
void l2_regularization(neural_network *, float);
void _init_adam_optimizer(Layer *);
void __adam_update(neural_network *, float, int);
float get_weight_gradient(neural_network *, size_t, size_t, size_t);

#endif
