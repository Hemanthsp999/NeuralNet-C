#include "neural.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

float assign_random_value(int threshold) {
        return (float)(rand() % (threshold + 1)) / threshold;
}

matrix *Multiplication(matrix mat1, matrix mat2) {

        if (mat1.cols != mat2.rows) {
                fprintf(stderr, "%s\n",
                        "Col of first matrix is not equal to rows of second "
                        "matrix");
                assert(mat1.cols != mat2.rows);
        }

        printf("Matrix 1 of Dimension: %ld\t Matrix 2 of Dimension: %ld\n",
               mat1.rows, mat2.rows);

        matrix *resultant = (matrix *)malloc(sizeof(matrix));
        resultant->rows = mat1.rows;
        resultant->cols = mat2.cols;

        resultant->data = (float **)malloc(resultant->rows * sizeof(float *));

        for (size_t i = 0; i < resultant->rows; i++) {
                resultant->data[i] =
                    (float *)calloc(resultant->cols, sizeof(float));
        }

        for (size_t i = 0; i < mat1.rows; i++) {
                for (size_t j = 0; j < mat2.cols; j++) {
                        resultant->data[i][j] = 0;
                        for (size_t k = 0; k < mat1.cols; k++) {
                                resultant->data[i][j] +=
                                    mat1.data[i][k] * mat2.data[k][j];
                        }
                }
        }

        return resultant;
}

matrix *Addition(matrix mat1, matrix mat2) {
        if (mat1.rows != mat2.rows || mat1.cols != mat2.cols) {
                fprintf(
                    stderr, "%s\n",
                    "Dimension of first matrix and second matrix are not same. "
                    "Both Matrix should be in same dimension for addition");

                assert(mat1.rows != mat2.rows || mat1.cols != mat2.cols);
        }

        matrix *resultant = (matrix *)malloc(sizeof(matrix));
        resultant->rows = mat1.rows;
        resultant->cols = mat1.cols;

        resultant->data = (float **)malloc(resultant->rows * sizeof(float *));

        for (size_t i = 0; i < resultant->rows; i++) {
                resultant->data[i] =
                    (float *)calloc(resultant->cols, sizeof(float));
        }

        for (size_t i = 0; i < mat1.rows; i++) {
                for (size_t j = 0; j < mat2.cols; j++) {
                        resultant->data[i][j] =
                            mat1.data[i][j] + mat2.data[i][j];
                }
        }

        return resultant;
}

matrix *Transpose(matrix mat1) {
        if (!mat1.rows || !mat1.cols) {
                fprintf(stderr, "%s\n",
                        "The size of Row or cols of given matrix is Zero");
                assert(!mat1.rows || !mat1.cols);
        }

        matrix *resultant = (matrix *)malloc(sizeof(matrix));
        resultant->rows = mat1.cols;
        resultant->cols = mat1.rows;

        resultant->data = (float **)malloc(resultant->rows * sizeof(float *));
        for (size_t i = 0; i < resultant->rows; i++) {
                resultant->data[i] =
                    (float *)malloc(resultant->cols * sizeof(float));
        }

        for (size_t i = 0; i < mat1.rows; i++) {
                for (size_t j = 0; j < mat1.cols; j++) {
                        resultant->data[j][i] = mat1.data[i][j];
                }
        }

        return resultant;
}

neural_network *Feed_Forward_Network(size_t *layer_size, size_t num_layers) {

        neural_network *network =
            (neural_network *)malloc(sizeof(neural_network));

        network->num_layers = num_layers;
        network->neural_layers = (Layer *)malloc(num_layers * sizeof(Layer));

        const int threshold = 1000;

        for (size_t i = 0; i < num_layers; i++) {
                network->neural_layers[i].num_neurons = layer_size[i];
                // Allocate memory for neurons
                network->neural_layers[i].neurons =
                    (Neuron *)malloc(layer_size[i] * sizeof(Neuron));

                // iterate over neurons in a layer
                for (size_t n = 0; n < layer_size[i]; n++) {
                        network->neural_layers[i].neurons[n].val = 0.f;

                        if (i < num_layers - 1) {
                                network->neural_layers[i].neurons[n].weight =
                                    (float *)malloc(layer_size[i + 1] *
                                                    sizeof(float));

                                /* assign value for each neuron; always check
                                 * for i+1 layer cause present neuron weight
                                 * combination is dependent on num of neurons in
                                 * next layer
                                 */
                                for (size_t w = 0; w < layer_size[i + 1]; w++) {
                                        // assign weight for each neuron
                                        network->neural_layers[i]
                                            .neurons[n]
                                            .weight[w] =
                                            assign_random_value(threshold);
                                }
                        } else {
                                // output layer don't have any weight. set it to
                                // null
                                network->neural_layers[i].neurons[n].weight =
                                    NULL;
                        }
                }
        }

        return network;
}

void forward_pass(neural_network *network, float *input, int bias) {

        //  add value to  input layer of neuron
        for (int i = 0; i < network->neural_layers[0].num_neurons; i++) {
                network->neural_layers[0].neurons[i].val = input[i];
        }

        for (size_t i = 0; i < network->num_layers - 1; i++) {

                Layer *current_layer = &network->neural_layers[i];
                Layer *next_layer = &network->neural_layers[i + 1];

                for (int j = 0; j < next_layer->num_neurons; j++) {
                        float sum = 0.f;
                        for (int k = 0; k < current_layer->num_neurons; k++) {
                                sum +=
                                    network->neural_layers[i].neurons[k].val *
                                    network->neural_layers[i]
                                        .neurons[k]
                                        .weight[j];
                        }
                        next_layer->neurons[j].val = sum + bias;
                }
        }
}
