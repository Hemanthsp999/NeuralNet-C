#include "../Neural/neural.h"
#include <stdio.h>
#ifndef FILE_
#define FILE_

typedef struct {
        float **x_train;
        float **x_test;
        int **y_train;
        int **y_test;
        float **X;
        int **Y;
        size_t samples;
        size_t train_samples;
        size_t test_samples;
        size_t input_features;
        size_t output_labels;
} dataset_handler;

dataset_handler *load_dataset(const char *);
dataset_handler *train_test_split(dataset_handler *, float, size_t);
dataset_handler *shuffle_dataset(float **, int **, float, size_t, size_t,
                                 size_t);
size_t count_columns(char *);
int *one_hot_encoder(const char *);
void train_network(neural_network *, dataset_handler *, const size_t);
void gradient_descent(float *);
void l2_regularization(int);

#endif
