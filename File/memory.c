#include "file.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* Deallocate the memory */
void free_dataset(dataset_handler *dataset) {
        if (!dataset) {
                fprintf(stderr, "Please pass Resource to Free.\n");
                assert(dataset);
        }

        if (dataset->x_val && dataset->y_val) {
                for (size_t i = 0; i < dataset->val_size; ++i) {
                        free(dataset->x_val[i]);
                        dataset->x_val[i] = NULL;
                        free(dataset->y_val[i]);
                        dataset->y_val[i] = NULL;
                }
                free(dataset->y_val);
                free(dataset->x_val);
        }

        if (dataset->x_test && dataset->y_test) {
                for (size_t i = 0; i < dataset->test_size; i++) {
                        free(dataset->x_test[i]);
                        dataset->x_test[i] = NULL;
                        free(dataset->y_test[i]);
                        dataset->y_test[i] = NULL;
                }
                free(dataset->y_test);
                free(dataset->x_test);
        }

        for (size_t i = 0; i < dataset->train_size; i++) {
                free(dataset->x_train[i]);
                dataset->x_train[i] = NULL;
                free(dataset->y_train[i]);
                dataset->y_train[i] = NULL;
        }
        free(dataset->x_train);
        free(dataset->y_train);

        for (size_t i = 0; i < dataset->samples; i++) {
                free(dataset->X[i]);
                dataset->X[i] = NULL;
        }
        free(dataset->X);
        dataset->X = NULL;
        free(dataset->Y);
        dataset->Y = NULL;

        free(dataset);
        dataset = NULL;
}

void free_network(NeuralNetwork *network) {
        if (!network) {
                fprintf(stderr, "Error while processing the Network.\n");
                return exit(EXIT_FAILURE);
        }

        size_t totalNetworkLayers = network->num_layers;
        for (size_t l = 0; l < totalNetworkLayers - 1; l++) {
                Layer *curr = &network->neural_layers[l];
                for (size_t j = 0; j < curr->num_neurons; j++) {
                        free(curr->neurons[j].weight);
                        curr->neurons[j].weight = NULL;
                }
                free(curr->neurons);
                curr->neurons = NULL;
        }

        free(network->neural_layers);
        network->neural_layers = NULL;
        free(network);
}
