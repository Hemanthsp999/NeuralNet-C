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
        float *weight;
} Neuron;

typedef struct {
        int num_neurons;
        Neuron *neurons; // holds the number of neurons of the one layer
} Layer;

typedef struct {
        size_t num_layers;
        Layer *neural_layers;
} neural_network;

matrix *Multiplication(matrix, matrix);
matrix *Addition(matrix, matrix);
matrix *Transpose(matrix);

neural_network *Feed_Forward_Network(size_t *, size_t);
void forward_pass(neural_network*, float*, int);
float assign_random_value(int);

#endif
