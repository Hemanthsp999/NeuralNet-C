#include <stdio.h>
#ifndef FILE_
#define FILE_

typedef struct {
        float **x_train;
        float **x_test;
        char **y_train;
        char **y_test;
        float **X;
        char **Y;
        size_t samples;
        size_t train_samples;
        size_t test_samples;
        size_t input_features;
        size_t output_labels;
} dataset_handler;

dataset_handler *load_dataset(const char *);
dataset_handler *train_test_split(dataset_handler *, float, size_t);
size_t count_columns(char *);
dataset_handler *shuffle_dataset(float **, char **, float, size_t, size_t,
                                 size_t);
void gradient_descent(float *);

#endif
