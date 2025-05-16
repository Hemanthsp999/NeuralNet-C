#include "file.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAX_LENGTH 1024

/* returns count of columns */
size_t count_columns(char *line) {
        size_t count = 1;
        while (*line) {
                if (*line == ',')
                        count++;

                line++;
        }
        return count;
}

/* returns dataset(X, Y) */
dataset_handler *load_dataset(const char *file_path) {
        FILE *file = fopen(file_path, "r");
        if (!file) {
                fprintf(stderr, "Error while opening the file\n");
                assert(!file);
        }

        if (!file_path) {
                fprintf(stderr, "The given file path is not correct\n");
                assert(!file_path);
                exit(1);
        }

        printf("The uploaded file: %s\n", file_path);

        if (fseek(file, 0, SEEK_END) != 0) {
                perror("Error, nothing in the file\n");
                fclose(file);
        }

        long file_size = ftell(file);

        if (file_size == -1L) {
                perror("Error in getting file size\n");
                exit(EXIT_FAILURE);
        }

        printf("The size of the dataset: %lu\n", file_size);

        fseek(file, 0, SEEK_SET);
        char line[MAX_LENGTH];
        printf("Line value: %d\n", *line);

        /* read the dataset and extract first row*/
        fgets(line, sizeof(line), file);
        int total_columns = count_columns(line);
        printf("Total number of Columns: %d\n", total_columns);
        size_t input_features = total_columns - 1;
        printf("Input feature: %ld\n", input_features);
        int output_labels = 1;
        printf("Output column: %d\n", output_labels);
        size_t samples = 1;

        while (fgets(line, sizeof(line), file)) {
                samples++;
        }

        printf("The count of samples: %ld\n", samples);
        // re-initialize curr position to the starting
        rewind(file);

        dataset_handler *handler =
            (dataset_handler *)malloc(sizeof(dataset_handler));
        if (!handler) {
                fprintf(stderr, "Error while allocating memory for handler.\n");
                assert(handler);
        }

        // initialize feature X to zero
        handler->X = (float **)malloc(samples * sizeof(float *));
        if (!handler->X) {
                fprintf(stderr, "Error while allocating memory for X and Y.\n");
                assert(handler->X);
        }
        // initialize output Y to zero
        handler->Y = (char **)malloc(samples * sizeof(char *));
        if (!handler->Y) {
                fprintf(stderr, "Error while allocating memory for X and Y.\n");
                assert(handler->Y);
        }
        /* Initialize other fields to null*/
        handler->x_train = NULL;
        handler->x_test = NULL;
        handler->y_train = NULL;
        handler->y_test = NULL;

        for (size_t i = 0; i < samples; i++) {
                handler->X[i] = (float *)calloc(input_features, sizeof(float));
                if (!handler->X[i]) {
                        fprintf(stderr, "Memory allocation failed at row %ld\n",
                                i);
                        exit(1);
                }

                handler->Y[i] = (char *)calloc(output_labels, sizeof(char));
                if (!handler->Y[i]) {
                        fprintf(
                            stderr,
                            "Memory allocation failed at output feature %ld\n",
                            i);
                        exit(1);
                }
        }

        size_t row = 0;
        while (fgets(line, sizeof(line), file)) {
                // parse csv
                char *token = strtok(line, ",");

                for (size_t col = 0; col < input_features; col++) {
                        handler->X[row][col] = atof(token);
                        token = strtok(NULL, ",");
                }

                handler->Y[row] = strdup(token);

                row++;
        }

        fclose(file);

        handler->input_features = input_features;
        handler->output_labels = output_labels;
        handler->samples = samples;
        return handler;
}

/* returns Train and Test dataset */
dataset_handler *train_test_split(dataset_handler *dataset, float test_size,
                                  size_t random_state) {
        if (!dataset) {
                fprintf(stderr, "The Provided Dataset is empty.\n");
                assert(!dataset);
        }

        // dataset_handler *shuffled_data =
        dataset =
            shuffle_dataset(dataset->X, dataset->Y, test_size, dataset->samples,
                            random_state, dataset->input_features);

        return dataset;
}

dataset_handler *shuffle_dataset(float **X, char **Y, float test_size,
                                 size_t samples, size_t random_state,
                                 size_t input_features) {
        if (!X) {
                fprintf(stderr, "The given data has no memory\n");
                exit(EXIT_FAILURE);
        }
        if (!Y) {
                fprintf(stderr, "The given data has no memory\n");
                exit(EXIT_FAILURE);
        }

        size_t total_samples = samples;
        size_t test_samples = (size_t)(total_samples * test_size);
        size_t train_samples = total_samples - test_samples;

        /* create a indices */
        size_t *indices = (size_t *)malloc(total_samples * sizeof(size_t));

        for (size_t i = 0; i < total_samples; i++) {
                indices[i] = i;
        }
        srand(random_state);

        /* shuffle the data and swap indices */
        for (int i = total_samples - 1; i > 0; i--) {
                size_t j = rand() % (i + 1);
                size_t tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
        }

        printf("Split size (Test size(20)): %f\n", test_size);

        dataset_handler *new_data =
            (dataset_handler *)malloc(sizeof(dataset_handler));

        /* Initialize all values to 0 */
        memset(new_data, 0, sizeof(dataset_handler));

        if (!new_data) {
                fprintf(stderr, "Memory allocation failed.\n");
                assert(new_data);
                exit(EXIT_FAILURE);
        }
        new_data->input_features = input_features;

        new_data->x_train = (float **)malloc(train_samples * sizeof(float *));
        new_data->y_train = (char **)malloc(train_samples * sizeof(char *));
        new_data->x_test = (float **)malloc(test_samples * sizeof(float *));
        new_data->y_test = (char **)malloc(test_samples * sizeof(char *));

        if (!new_data->x_train || !new_data->y_train || !new_data->x_test ||
            !new_data->y_test) {
                fprintf(stderr, "Memory allocation failed for training/testing "
                                "data. source: shuffle_dataset()\n");
                exit(EXIT_FAILURE);
        }

        for (size_t i = 0; i < train_samples; i++) {
                new_data->x_train[i] =
                    (float *)calloc(input_features, sizeof(float));
                for (size_t j = 0; j < input_features; j++) {
                        new_data->x_train[i][j] = X[indices[i]][j];
                }
                /* copy entire Y pointer to y_train */
                new_data->y_train[i] = strdup(Y[indices[i]]);
        }

        for (size_t i = 0; i < test_samples; i++) {
                new_data->x_test[i] =
                    (float *)calloc(input_features, sizeof(float));
                for (size_t j = 0; j < input_features; j++) {
                        new_data->x_test[i][j] =
                            X[indices[train_samples + i]][j];
                }
                /* copy entire Y pointer to y_test */
                new_data->y_test[i] = strdup(Y[indices[train_samples + i]]);
        }
        /*
            new_data->X = (float **)malloc(samples * sizeof(float));
            new_data->Y = (char **)malloc(samples * sizeof(char));

            for (size_t i = 0; i < samples; i++) {
                    new_data->X[i] = calloc(input_features, sizeof(float));
                    for (size_t j = 0; j < input_features; j++) {
                            new_data->X[i][j] = X[i][j];
                    }

                    new_data->Y[i] = strdup(Y[i]);
            }
        */
        new_data->samples = samples;
        new_data->train_samples = train_samples;
        new_data->test_samples = test_samples;

        free(indices);
        return new_data;
}
