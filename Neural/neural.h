/* neural.h */
#include <stdio.h>
#ifndef Neural_
#define Neural_

#define Dim(a) (sizeof(a) / sizeof(a[0]))

typedef struct {
        float **data;
        size_t rows;
        size_t cols;
} matrix;

typedef struct {
        float val;
        float bias;
        float deltas;
        float *weight;
} Neuron;

typedef struct {
        size_t num_neurons; // holds the number of neurons in the layer
        Neuron *neurons;    // holds neuron val and weight
} Layer;

typedef struct {
        size_t num_layers;
        Layer *neural_layers; /* is a pointer to array, helps to trace the layer
                                     wise neurons and perform forwardpass */
} neural_network;

matrix *Multiplication(matrix, matrix);
matrix *Addition(matrix, matrix);
matrix *Transpose(matrix);

neural_network *Feed_Forward_Network(size_t *, size_t);
void back_propagation(neural_network *, int *, Layer *);
void forward_pass(neural_network *, float *);
float assign_random_value(int);

#endif
