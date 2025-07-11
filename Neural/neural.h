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

/* Chain link structure */
typedef struct {
        float val;
        float bias;
        float delta;
        float *weight;
} Neuron;

typedef struct {
        size_t num_neurons; // -> holds the number of neurons in the layer
        Neuron *neurons;    // -> holds neuron val, weight, bias and delta

        /* Adam optimizer handler*/
        float **m_w; // momentum weights
        float *m_b;  // momentum bias
        float **v_w; // velocity weights
        float *v_b;  // velocity bias
} Layer;

typedef struct {
        size_t num_layers;
        Layer *neural_layers; /* is a pointer, helps to trace the layer
                                     wise neurons */
} NeuralNetwork;

matrix *_Multiplication(matrix, matrix);
matrix *_Addition(matrix, matrix);
matrix *_Transpose(matrix);

NeuralNetwork *Feed_Forward_Network(size_t *, size_t);
void back_propagation(NeuralNetwork *, int *, Layer *);
void forward_pass(NeuralNetwork *, float *, _Bool);
float assign_random_value(int);

#endif
