/* neural.c */
#include "neural.h"
#include "../Activation/activation.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* retuns random value between 0 to 1000 */
float assign_random_value(int threshold) {
        return (float)(rand() % (threshold + 1)) / threshold;
}

/* returns dot product */
matrix *_Multiplication(matrix mat1, matrix mat2) {

        if (mat1.cols != mat2.rows) {
                fprintf(stderr, "%s\n",
                        "Col of first matrix is not equal to rows of second "
                        "matrix");
                exit(EXIT_FAILURE);
        }

        printf("Matrix 1 of Row: %ld\t Cols: %ld\t Matrix 2 of Row: %ld\t "
               "Cols: %ld\n",
               mat1.rows, mat1.cols, mat2.rows, mat2.cols);

        matrix *resultant = (matrix *)malloc(sizeof(matrix));

        if (!resultant) {
                fprintf(stderr,
                        "Memory is not allocated. Source: Multiplication()\n");
                free(resultant);
                assert(!resultant);
        }
        resultant->rows = mat1.rows;
        resultant->cols = mat2.cols;

        resultant->data = (float **)malloc(resultant->rows * sizeof(float *));
        if (!resultant->data) {
                fprintf(stderr,
                        "Memory is not allocated. Source: Multiplication()\n");
                free(resultant->data);
                free(resultant);
                assert(!resultant->data);
        }

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

matrix *_Addition(matrix mat1, matrix mat2) {
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

        if (!resultant) {
                fprintf(stderr, "Memory is not allocated. Source: Addition()");
                free(resultant->data);
                free(resultant);
                assert(!resultant);
        }

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

/* returns transpose of a matrix i.e [1 2] -> [2 1] */
matrix *_Transpose(matrix mat1) {
        if (!mat1.rows || !mat1.cols) {
                fprintf(stderr, "%s\n",
                        "The size of Row or cols of given matrix is Zero\n");
                assert(!mat1.rows || !mat1.cols);
        }

        matrix *resultant = (matrix *)malloc(sizeof(matrix));
        if (!resultant) {
                fprintf(stderr,
                        "Memory is not allocated. Source: Transpose()\n");
                free(resultant);
                assert(!resultant);
        }

        resultant->rows = mat1.cols;
        resultant->cols = mat1.rows;

        resultant->data = (float **)malloc(resultant->rows * sizeof(float *));
        if (!resultant->data) {
                fprintf(stderr,
                        "Memory is not allocated. Source: Transpose()\n");
                free(resultant->data);
                free(resultant);
                assert(!resultant->data);
                assert(!resultant);
        }

        for (size_t i = 0; i < resultant->rows; i++) {
                resultant->data[i] =
                    (float *)calloc(resultant->cols, sizeof(float));
        }

        for (size_t i = 0; i < mat1.rows; i++) {
                for (size_t j = 0; j < mat1.cols; j++) {
                        resultant->data[j][i] = mat1.data[i][j];
                }
        }

        return resultant;
}

/* returns network with connected neurals and weights assigned for it  */
NeuralNetwork *Feed_Forward_Network(size_t *neuron_in_layers,
                                    size_t num_layers) {
        if (!neuron_in_layers) {
                fprintf(stderr, "The layer Size is empty. \n");
                assert(!neuron_in_layers);
        }

        NeuralNetwork *network = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));

        if (!network) {
                fprintf(stderr, "Memory allocation failed. Source: "
                                "Feed_Forward_Network()\n");
                free(network);
                assert(!network);
        }
        printf("Now the function is in Feed_forward_network\n");

        network->num_layers = num_layers;
        network->neural_layers = (Layer *)malloc(num_layers * sizeof(Layer));
        if (!network->neural_layers) {
                fprintf(
                    stderr,
                    "Memory allocation failed. Source: Feed_Forward_Network\n");
                free(network->neural_layers);
                assert(!network->neural_layers);
        }

        const int threshold = 1000;

        printf("Number of layers present in Neural Network: %ld\n", num_layers);

        for (size_t i = 0; i < num_layers; i++) {
                printf("Number of Neurons present in Layer[%ld]: %ld\n", i,
                       neuron_in_layers[i]);
        }

        for (size_t i = 0; i < num_layers; i++) {
                network->neural_layers[i].num_neurons = neuron_in_layers[i];

                // Allocate memory for neurons and initalizes to zero
                network->neural_layers[i].neurons =
                    (Neuron *)calloc(neuron_in_layers[i], sizeof(Neuron));

                /* check if memory is allocated for neurons */
                if (!network->neural_layers[i].neurons) {
                        fprintf(stderr,
                                "Error while allocating memory for neurons. "
                                "Source: Feed_Forward_Network().135");
                        free(network->neural_layers[i].neurons);
                        assert(!network->neural_layers[i].neurons);
                }

                Layer *curr_layer = &network->neural_layers[i];
                /* iterate over neurons in a layer */
                for (size_t n = 0; n < neuron_in_layers[i]; n++) {
                        curr_layer->neurons[n].val = 0.f;

                        if (i < num_layers - 1) {
                                /* Initialize weights to zero */
                                network->neural_layers[i].neurons[n].weight =
                                    (float *)calloc(neuron_in_layers[i + 1],
                                                    sizeof(float));

                                /* assign value for each neuron; always check
                                 * for i+1 layer cause present neuron weight
                                 * combination is dependent on num of neurons in
                                 * next layer
                                 */
                                for (size_t w = 0; w < neuron_in_layers[i + 1];
                                     w++) {
                                        // assign weight for each neuron
                                        curr_layer->neurons[n].weight[w] =
                                            assign_random_value(threshold);
                                }
                        } else {
                                /* output layer don't have any weights.
                                 * Therefore it will be NULL */
                                curr_layer->neurons[n].weight = NULL;
                        }
                }
        }

        /* Extrad Added */
        for (size_t i = 1; i < num_layers; i++) {
                Layer *curr_layer = &network->neural_layers[i];
                for (size_t j = 0; j < curr_layer->num_neurons; j++) {
                        curr_layer->neurons[j].bias =
                            assign_random_value(threshold);
                }
        }

        return network;
}

/* return activated networks */
void forward_pass(NeuralNetwork *network, float *input, _Bool debug) {

        if (!network) {
                fprintf(stderr, "%s\n", "Network is empty?.");
                free(network);
                assert(!network);
        }
        printf("Now the function is in forward_Pass\n");

        /*  Initialize values to  input layer of neuron */
        size_t num_input_neurons = network->neural_layers[0].num_neurons;
        for (size_t i = 0; i < num_input_neurons; i++) {
                network->neural_layers[0].neurons[i].val = input[i];
        }

        size_t total_layers = network->num_layers;

        // activate neurons
        for (size_t i = 0; i < total_layers - 1; ++i) {
                // printf("Processing the forward pass: %ld ...\n", i + 1);

                Layer *current_layer = &network->neural_layers[i];
                Layer *next_layer = &network->neural_layers[i + 1];

                for (size_t j = 0; j < next_layer->num_neurons; ++j) {
                        float sum = 0.f;

                        for (size_t k = 0; k < current_layer->num_neurons;
                             ++k) {
                                float curr_neuron_val =
                                    current_layer->neurons[k].val;
                                float curr_neuron_weight =
                                    current_layer->neurons[k].weight[j];

                                if (debug)
                                        printf("Weights [%zu][%zu]: %f \n", k,
                                               j, curr_neuron_weight);

                                sum += curr_neuron_val * curr_neuron_weight;
                        }
                        if (debug)
                                printf("\n");

                        // pass sum with bias to sigmoid function
                        sum += next_layer->neurons[j].bias;

                        // assign calculated value to next layer neurons
                        next_layer->neurons[j].val = sigmoid(sum);
                }
        }

        // pass values to softmax func()
        soft_max(&network->neural_layers[network->num_layers - 1]);
}

/* Calculate Delta values excluding input layer and updates weights and biases
 */
void back_propagation(NeuralNetwork *network, int *output_expected,
                      Layer *output_predicted) {
        if (!network) {
                fprintf(stderr, "the network is empyt.\n");
                free(network);
                return exit(EXIT_FAILURE);
        }

        if (!output_expected) {
                fprintf(stderr, "The Expected output values are empyt.\n");
                free(output_expected);
                return exit(EXIT_FAILURE);
        }

        if (!output_predicted) {
                fprintf(stderr, "The Predicted output values are empyt.\n");
                free(output_predicted);
                return exit(EXIT_FAILURE);
        }

        printf("Debug: Now the function is in Back_Propagation\n");

        // number of layers that network contains
        size_t n_layers = network->num_layers;

        // points to last layer
        size_t output_layer = n_layers - 1;
        size_t output_layer_neurons =
            network->neural_layers[output_layer].num_neurons;

        // calculate delta value for output layer
        Layer *outputLayer = &network->neural_layers[output_layer];
        for (size_t i = 0; i < output_layer_neurons; i++) {
                float a = output_predicted->neurons[i].val;
                float y = (float)output_expected[i];
                float z_derivative = sigmoid_derivative(a);

                float delta_value = (a - y) * z_derivative;

                outputLayer->neurons[i].delta = delta_value;

                printf("Output neuron %zu: value=%f, expected=%f, delta=%f\n",
                       i, a, y, output_predicted->neurons[i].delta);
        }

        // Backpropagate delta
        for (int l = output_layer - 1; l > 0; --l) {
                Layer *current_layer = &network->neural_layers[l];
                Layer *next_layers = &network->neural_layers[l + 1];

                for (size_t i = 0; i < current_layer->num_neurons; i++) {
                        float error_sum = 0.f;
                        for (size_t j = 0; j < next_layers->num_neurons; j++) {
                                error_sum +=
                                    next_layers->neurons[j].delta *
                                    current_layer->neurons[i].weight[j];
                        }

                        float curr_neuron_val = current_layer->neurons[i].val;
                        float z_derivative =
                            sigmoid_derivative(curr_neuron_val);

                        current_layer->neurons[i].delta =
                            error_sum * z_derivative;
                }
        }

        float learning_rate = 0.001f;

        // update weight
        for (size_t l = 0; l < output_layer; l++) {
                Layer *current_layer = &network->neural_layers[l];
                Layer *next_layer = &network->neural_layers[l + 1];

                for (size_t i = 0; i < next_layer->num_neurons; i++) {

                        for (size_t n = 0; n < current_layer->num_neurons;
                             n++) {
                                /* Update weights: weight -= learning_rate *
                                 * input_activation * delta
                                 */

                                float current_neuron_value =
                                    current_layer->neurons[n].val;

                                float next_layer_delta =
                                    next_layer->neurons[i].delta;

                                current_layer->neurons[n].weight[i] -=
                                    learning_rate * current_neuron_value *
                                    next_layer_delta;
                        }
                }
        }

        /* update bias */
        for (size_t l = 1; l < n_layers; l++) {
                /* layer always starts from 1 cause 0 is
                 * input layer and input layer don't
                 * have any bias */
                Layer *curr_layer = &network->neural_layers[l];

                /* formula to update bias: new_bias =
                 * old_bias * lr * delta */
                for (size_t i = 0; i < curr_layer->num_neurons; i++) {

                        curr_layer->neurons[i].bias -=
                            learning_rate * curr_layer->neurons[i].delta;
                }
        }

        /*
            for (size_t i = 0; i < network->num_layers; i++) {

                    for (size_t j = 0; j <
           network->neural_layers[i].num_neurons; j++) { printf("\nNeuron Value:
           %f ", network->neural_layers[i].neurons[j].val);
                    }
                    printf("\n");
            }
        */
}
