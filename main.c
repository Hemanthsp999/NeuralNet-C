#include "File/file.h"
#include "File/memory.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
        printf("Neural Network Library in C, still work in progress\n");

        /* load dataset */
        const char *file_name = "iris.csv";
        dataset_handler *dataset = load_dataset(file_name);

        for (size_t i = 0; i < dataset->samples; i++) {
                printf("Sample %ld: ", i + 1);
                for (size_t j = 0; j < dataset->input_features; j++) {
                        printf("%f ", dataset->X[i][j]);
                }
                for (size_t k = 0; k < 3; k++) {
                        printf("%d ", dataset->Y[i][k]);
                }

                printf("\n");
        }

        float test_size = 0.2f;
        size_t random_state = 42;
        dataset_handler *split_data =
            train_test_split(dataset, test_size, random_state);

        printf("The Train samples\n");
        for (size_t i = 0; i < split_data->train_samples; i++) {
                printf("Train samples: %ld  ", i + 1);
                for (size_t j = 0; j < split_data->input_features; j++) {
                        printf("%f ", split_data->x_train[i][j]);
                }
                for (int k = 0; k < 3; k++) {
                        printf("%d ", split_data->y_train[i][k]);
                }
                printf("\n");
        }

        printf("These are the Testing samples\n");
        for (size_t i = 0; i < split_data->test_samples; i++) {
                printf("Test samples: %ld ", i + 1);
                for (size_t j = 0; j < split_data->input_features; j++) {
                        printf("%f ", split_data->x_test[i][j]);
                }
                for (int k = 0; k < 3; k++) {
                        printf("%d ", split_data->y_test[i][k]);
                }
                printf("\n");
        }
        printf("The train samples: %ld\n", split_data->train_samples);
        printf("The test samples: %ld\n", split_data->test_samples);
        printf("The total samples: %ld\n", split_data->samples);

        size_t input_values[] = {dataset->input_features, 4, 4, 3};
        size_t *layer_values = input_values;

        neural_network *construct_network =
            Feed_Forward_Network(layer_values, 4);

        train_network(construct_network, dataset, 1000);

        /*
            free_dataset(dataset);
            dataset = NULL;
            dataset = split_data;
            split_data->X = NULL;
            split_data->Y = NULL;
            free_dataset(split_data);
        */

        split_data->X = NULL;
        split_data->Y = NULL;
        free_dataset(split_data);
        split_data = NULL;
        free_dataset(dataset);
        dataset = NULL;

        return 0;
}
