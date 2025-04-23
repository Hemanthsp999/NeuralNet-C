#ifndef Neural_
#define Neural_

#define Dim(a) (sizeof(a) / sizeof(a[0]))

typedef struct {
        float val;
        float *weight;
} Neuron;

typedef struct {
        float **data;
        int rows;
        int cols;
} matrix;

typedef struct {
        int input_layers;
        int hidden_layers;
        int output_layers;
} Layers;

/*
typedef struct {
        float **input_to_hidden_weights;
        float **hidden_to_output_weights;
        float **input_to_hidden_bias;
        float **hidden_to_output_bias;
} weight_bias_map;
*/

matrix *Matrix_Multiplication(matrix, matrix);
matrix *Matrix_Addition(matrix, matrix);
matrix *Transpose(matrix);

Neuron *input_neuron_weights(Neuron *, Layers);
Neuron *Feed_Forward_Network(Neuron *, Layers, int);

#endif
