/* file.h */
#include "../Neural/neural.h"
#include <stdio.h>
#ifndef FILE_
#define FILE_

typedef struct {
        float **x_train;
        float **x_test;
        float **x_val;
        int **y_train;
        int **y_test;
        int **y_val;

        float **X; // -> X feature
        int **Y;   // -> Y label

        size_t samples;
        size_t train_size;
        size_t test_size;
        size_t val_size;
        size_t input_features;
        size_t output_labels;
        size_t total_output_class;

} dataset_handler;

dataset_handler *load_dataset(const char *, size_t);
dataset_handler *train_test_split(dataset_handler *, float, size_t);
dataset_handler *shuffle_dataset(float **, int **, float, size_t, size_t,
                                 size_t);
size_t count_columns(char *);
size_t get_output_labels(char *);
int *one_hot_encoder(const char *, size_t);
char *one_hot_decoder(int x);
NeuralNetwork *load_model(NeuralNetwork *, const char *);
void _train_network(NeuralNetwork *, dataset_handler *, const size_t, _Bool);
void gradient_descent(float *);
void display_network(NeuralNetwork *);
void _save_model_(NeuralNetwork *, const char *);
void predict_(NeuralNetwork *, float **, int **, size_t, size_t, size_t, _Bool);
void validate_network(NeuralNetwork *, float **, int **, size_t, size_t, _Bool);

#endif
