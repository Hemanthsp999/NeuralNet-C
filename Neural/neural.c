#include "neural.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

matrix *Matrix_Multiplication(matrix mat1, matrix mat2) {

        if (mat1.cols != mat2.rows) {
                fprintf(stderr, "%s\n",
                        "Col of first matrix is not equal to rows of second "
                        "matrix");
                assert(mat1.cols != mat2.rows);
        }

        printf("Matrix 1 of Dimension: %d\t Matrix 2 of Dimension: %d\n",
               mat1.rows, mat2.rows);

        matrix *resultant = (matrix *)malloc(sizeof(matrix));
        resultant->rows = mat1.rows;
        resultant->cols = mat2.cols;

        resultant->data = (float **)malloc(resultant->rows * sizeof(float *));

        for (int i = 0; i < resultant->rows; i++) {
                resultant->data[i] =
                    (float *)calloc(resultant->cols, sizeof(float));
        }

        for (int i = 0; i < mat1.rows; i++) {
                for (int j = 0; j < mat2.cols; j++) {
                        resultant->data[i][j] = 0;
                        for (int k = 0; k < mat1.cols; k++) {
                                resultant->data[i][j] +=
                                    mat1.data[i][k] * mat2.data[k][j];
                        }
                }
        }

        return resultant;
}

matrix *Matrix_Addition(matrix mat1, matrix mat2) {
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

        for (int i = 0; i < resultant->rows; i++) {
                resultant->data[i] =
                    (float *)calloc(resultant->cols, sizeof(float));
        }

        for (int i = 0; i < mat1.rows; i++) {
                for (int j = 0; j < mat2.cols; j++) {
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
        for (int i = 0; i < resultant->rows; i++) {
                resultant->data[i] =
                    (float *)malloc(resultant->cols * sizeof(float));
        }

        for (int i = 0; i < mat1.rows; i++) {
                for (int j = 0; j < mat1.cols; j++) {
                        resultant->data[j][i] = mat1.data[i][j];
                }
        }

        return resultant;
}

Neuron *input_neuron_weights(Neuron *neuron, Layers layer) {

        const int threshold = 1000;

        if (neuron == NULL) {
                fprintf(stderr, "%s\n",
                        "Memory is not allocated for input_neuron");
                assert(neuron == NULL);
                exit(EXIT_FAILURE);
        }

        /* input neuron weights is direclty proportional to hidden_layer neurons
         */
        neuron->weight = (float *)malloc(layer.hidden_layers * sizeof(float));

        for (int i = 0; i < layer.hidden_layers; i++) {
                neuron->weight[i] =
                    (float)((rand()) % (threshold + 1)) / threshold;
        }

        return neuron;
}

Neuron *Feed_Forward_Network(Neuron *input_neuron, Layers layer, int bias) {
        if (!input_neuron) {
                fprintf(stderr, "%s\n",
                        "Weight of hidden/input neuron is Zero");
                assert(!input_neuron->weight);
        }
        /* FNN(x) = x.weight+bias*/
        const int threshold = 1000;

        Neuron *hidden_neuron = malloc(layer.hidden_layers * sizeof(Neuron));
        hidden_neuron->weight = (float *)malloc(sizeof(float));

        for (int i = 0; i < layer.hidden_layers; i++) {
                hidden_neuron[i].weight =
                    (float *)malloc(layer.hidden_layers * sizeof(float));
        }

        for (int i = 0; i < layer.hidden_layers; i++) {
                float sum = 0.f;
                for (int j = 0; j < layer.input_layers; j++) {
                        sum += input_neuron[i].val * input_neuron[i].weight[j];
                }

                hidden_neuron[i].val = sum + bias;
                hidden_neuron[i].weight[i] =
                    (float)((rand()) % (threshold + 1)) / threshold;
        }

        return hidden_neuron;
}
