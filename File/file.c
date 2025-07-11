#include "file.h"
#include "Activation/activation.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAX_LENGTH 1024
#define MAX_LABEL_CLASS 200

/* returns total output labels from dataset */
size_t get_output_labels(char *fileName) {
        if (!fileName) {
                fprintf(stderr, "Error: Given file is Either corrupted or "
                                "something went wrong\n");
                return -1;
        }

        FILE *file = fopen(fileName, "r");
        if (!fileName) {
                fprintf(stderr, "Error while opening the file.\n");
                return -1;
        }

        char line[MAX_LENGTH];
        int label_count = 0;
        char *unique_labels[MAX_LABEL_CLASS];
        fgets(line, sizeof(line), file);
        while (fgets(line, sizeof(line), file)) {

                line[strcspn(line, "\r\n")] = 0;
                char *token = strtok(line, ",");
                char *last_token = NULL;

                // track the last label and store it in last_token
                while (token) {
                        last_token = token;
                        token = strtok(NULL, ",");
                }

                int found = 0;
                // check if label is already present in unique_labels set
                for (int i = 0; i < label_count; i++) {
                        if (strcmp(unique_labels[i], last_token) == 0) {
                                found = 1;
                                break;
                        }
                }

                if (!found && last_token) {
                        unique_labels[label_count] = strdup(last_token);
                        label_count++;
                }
        }

        for (int i = 0; i < label_count; i++) {
                printf("Debug output label: %s\n", unique_labels[i]);
        }
        printf("Total labels: %d\n", label_count);

        fclose(file);
        return label_count;
}

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

/* Decodes Integer Class to Output label*/
char *one_hot_decoder(int x) {

        char *label_string = "";
        if (x == 0) {
                label_string = "Setosa";
        } else if (x == 1) {
                label_string = "Versicolor";
        } else if (x == 2) {
                label_string = "Virginica";
        }

        return label_string;
}

/* Encodes Output Label to Integer Class i.e: {"a", "b"} -> {0, 1} */
int *one_hot_encoder(const char *string, size_t num_classes) {
        if (!string) {
                fprintf(stderr, "Please enter a valid string: %s\n", string);
                exit(EXIT_FAILURE);
        }

        int *encoder = (int *)malloc(num_classes * sizeof(int));
        if (!encoder) {
                fprintf(stderr,
                        "Error while allocating memory for Encoder block.\n");
                assert(encoder);
        }

        for (int i = 0; i < (int)num_classes; i++) {
                encoder[i] = 0;
        }

        char *trimmed_string = strdup(string);

        // Trim whitespace and newlines
        size_t len = strlen(trimmed_string);
        while (len > 0 && (trimmed_string[len - 1] == '\n' ||
                           trimmed_string[len - 1] == '\r' ||
                           trimmed_string[len - 1] == ' ' ||
                           trimmed_string[len - 1] == '"')) {
                trimmed_string[--len] = '\0';
        }

        // Remove leading whitespace or quotes
        char *start = trimmed_string;
        while (*start == ' ' || *start == '"') {
                start++;
        }

        // Debug print
        printf("Processing label: '%s'\n", start);

        if (strcmp(start, "Setosa") == 0) {
                encoder[0] = 1;
        } else if (strcmp(start, "Versicolor") == 0) {
                encoder[1] = 1;
        } else if (strcmp(start, "Virginica") == 0) {
                encoder[2] = 1;
        } else {
                printf("Warning: Unknown class '%s'\n", start);
        }

        free(trimmed_string);
        return encoder;
}

/* returns dataset -> X feature | Y labels*/
dataset_handler *load_dataset(const char *file_path, size_t num_output_class) {
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

        int output_cols = total_columns - input_features;
        printf("Output label: %d\n", output_cols);
        size_t samples = 1;

        // count total size of data inside dataset
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
        handler->Y = (int **)malloc(samples * sizeof(int *));
        if (!handler->Y) {
                fprintf(stderr, "Error while allocating memory for X and Y.\n");
                assert(handler->Y);
        }

        /* Initialize other fields to null*/
        handler->x_train = NULL;
        handler->x_test = NULL;
        handler->x_val = NULL;
        handler->y_val = NULL;
        handler->y_train = NULL;
        handler->y_test = NULL;

        /* time complexity -> O(n)*/
        for (size_t i = 0; i < samples; i++) {
                handler->X[i] = (float *)calloc(input_features, sizeof(float));
                if (!handler->X[i]) {
                        fprintf(stderr, "Memory allocation failed at row %ld\n",
                                i);
                        exit(1);
                }

                handler->Y[i] = (int *)calloc(output_cols, sizeof(int));
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

                handler->Y[row] = one_hot_encoder(token, num_output_class);

                row++;
        }

        fclose(file);

        handler->input_features = input_features;
        handler->output_labels = output_cols;
        printf("OutputLabels: %d\n", output_cols);
        handler->samples = samples;
        handler->total_output_class = num_output_class;
        return handler;
}

/* returns Train, Val and Test sets */
dataset_handler *train_test_split(dataset_handler *dataset, float test_size,
                                  size_t random_state) {
        if (!dataset) {
                fprintf(stderr, "The Provided Dataset is empty.\n");
                assert(!dataset);
        }

        dataset =
            shuffle_dataset(dataset->X, dataset->Y, test_size, dataset->samples,
                            random_state, dataset->input_features);

        return dataset;
}

/* returns randomly shuffled train, val and test data */
dataset_handler *shuffle_dataset(float **X, int **Y, float test_size,
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
        size_t test_val_samples = (size_t)(total_samples * test_size);
        size_t test_samples = test_val_samples / 2;
        size_t validation_size = test_val_samples / 2;
        size_t train_samples = total_samples - test_val_samples;

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

        float train_percent = ((float)train_samples / total_samples);
        float test_percent = ((float)test_samples / total_samples);
        float val_percent = ((float)validation_size / total_samples);

        printf("Split: Train(%f Percent): %zu\t Validation(%f Percent): %zu\t "
               "Test(%f Percent): %zu\n",
               round(100.0 * train_percent), train_samples,
               round(val_percent * 100), validation_size,
               round(test_percent * 100), test_samples);

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

        new_data->X = (float **)malloc(samples * sizeof(float *));
        new_data->Y = (int **)malloc(samples * sizeof(int *));
        new_data->x_train = (float **)malloc(train_samples * sizeof(float *));
        new_data->y_train = (int **)malloc(train_samples * sizeof(int *));
        new_data->x_test = (float **)malloc(test_samples * sizeof(float *));
        new_data->y_test = (int **)malloc(test_samples * sizeof(int *));
        new_data->x_val = (float **)malloc(validation_size * sizeof(float *));
        new_data->y_val = (int **)malloc(validation_size * sizeof(int *));

        if (!new_data->x_train || !new_data->y_train || !new_data->x_test ||
            !new_data->y_test || !new_data->x_val || !new_data->y_val) {
                fprintf(stderr, "Memory allocation failed for training/testing "
                                "data. source: shuffle_dataset()\n");
                exit(EXIT_FAILURE);
        }

        for (size_t i = 0; i < samples; i++) {
                new_data->X[i] = calloc(input_features, sizeof(float));
                for (size_t j = 0; j < input_features; j++) {
                        new_data->X[i][j] = X[i][j];
                }
                new_data->Y[i] = Y[i];
        }

        /* Training Set */
        for (size_t i = 0; i < train_samples; i++) {
                new_data->x_train[i] =
                    (float *)calloc(input_features, sizeof(float));
                for (size_t j = 0; j < input_features; j++) {
                        new_data->x_train[i][j] = X[indices[i]][j];
                }
                new_data->y_train[i] = Y[indices[i]];
        }

        /* Validation Set */
        for (size_t i = 0; i < validation_size; i++) {
                new_data->x_val[i] =
                    (float *)calloc(input_features, sizeof(float));
                for (size_t j = 0; j < input_features; j++) {
                        new_data->x_val[i][j] =
                            X[indices[train_samples + i]][j];
                }
                new_data->y_val[i] = Y[indices[train_samples + i]];
        }

        /* Testing Set */
        for (size_t i = 0; i < test_samples; i++) {
                new_data->x_test[i] =
                    (float *)calloc(input_features, sizeof(float));
                for (size_t j = 0; j < input_features; j++) {
                        new_data->x_test[i][j] =
                            X[indices[train_samples + validation_size + i]][j];
                }
                /* copy entire Y pointer to y_test */
                // new_data->y_test[i] = strdup(Y[indices[train_samples + i]]);
                new_data->y_test[i] =
                    Y[indices[train_samples + validation_size + i]];
        }

        new_data->samples = samples;
        new_data->train_size = train_samples;
        new_data->val_size = validation_size;
        new_data->test_size = test_samples;

        free(indices);
        return new_data;
}

void _train_network(NeuralNetwork *network, dataset_handler *dataset,
                    const size_t epochs, _Bool debug) {
        if (!network || !dataset) {
                fprintf(stderr,
                        "The network is empty or the dataset is NULL.\n");
                exit(EXIT_FAILURE);
        }

        printf("Input Features: %ld train_samples: %ld \n",
               dataset->input_features, dataset->train_size);

        printf("Starting training with %ld epochs\n", epochs);
        size_t num_classes = dataset->total_output_class;

        _init_adam_optimizer(network);
        float learning_rate = 0.05f;
        int time_step = 0;

        for (size_t epoch = 0; epoch < epochs; epoch++) {
                printf("\n=== Processing epoch: %ld/%ld ===\n", epoch + 1,
                       epochs);
                float total_error = 0.0f;

                for (size_t i = 0; i < dataset->train_size; i++) {
                        time_step++;
                        // Extract input features for this sample
                        float input_features[dataset->input_features];
                        for (size_t j = 0; j < dataset->input_features; j++) {
                                input_features[j] = dataset->x_train[i][j];
                        }

                        // Extract expected outputs for this sample
                        int output_expected[num_classes];
                        for (size_t t = 0; t < num_classes; t++) {
                                output_expected[t] = dataset->y_train[i][t];
                        }

                        // Log only every 10th sample to avoid console flood
                        if (i % 10 == 0) {
                                printf("Processing sample %zu/%zu...\n", i + 1,
                                       dataset->train_size);
                        }

                        // Forward pass

                        forward_pass(network, input_features, debug);

                        // Calculate error for logging (mean squared error)
                        /*
                                    float sample_error = 0.0f;
                                    for (size_t t = 0; t < num_classes; t++) {
                                            float err =
                                                predicted_output
                                                    ->neural_layers[network->num_layers
                           - 1] .neurons[t] .val - (float)output_expected[t];
                                            sample_error += err * err;
                                    }
                                    sample_error /= num_classes;
                        */

                        float mse_error = mse(
                            &network->neural_layers[network->num_layers - 1],
                            (float *)output_expected);
                        total_error += mse_error;

                        // Back propagation - pass the whole network and specify
                        // we want to update its last layer
                        back_propagation(
                            network, output_expected,
                            &network->neural_layers[network->num_layers - 1]);
                        l2_regularization(network, 0.4f);
                        // display_network(predicted_output);
                        __adam_update(network, learning_rate, time_step);
                }

                printf("......................................................."
                       "...................................\n");
                printf("------------------------------> Average Error: %f "
                       "<--------------------------------\n",
                       total_error);
        }

        printf("Training completed!\n");

        _save_model_(network, "model_weights.txt");
}

void display_network(NeuralNetwork *network) {
        if (!network) {
                fprintf(stderr, "Error\n");
                assert(network);
        }

        printf("Displaying Neural Network\n");
        for (size_t i = 0; i < network->num_layers - 1; i++) {
                Layer *curr = &network->neural_layers[i];
                for (size_t j = 0; j < curr->num_neurons; j++) {
                        printf(" ( %f ) ", curr->neurons[j].val);
                }
                printf("\n");
        }
}

void _save_model_(NeuralNetwork *network, const char *file_name) {

        if (!network) {
                fprintf(stderr, "Error: The network is not constructed.\n");
                exit(EXIT_FAILURE);
        }

        if (!file_name) {
                fprintf(stderr, "Please enter the file name\n");
                assert(file_name);
        }

        FILE *file = fopen(file_name, "w");
        if (!file) {
                fprintf(stderr, "Error: could not open file\n");
                return exit(EXIT_FAILURE);
        }

        int layer_size = network->num_layers;

        fprintf(file, "Total Layers: %d\n", layer_size);
        for (size_t i = 0; i < network->num_layers; i++) {
                fprintf(file, "Layer[%ld]: %ld\n", i,
                        network->neural_layers[i].num_neurons);
        }

        for (int i = 0; i < 60; i++) {
                fprintf(file, "*");
        }
        fprintf(file, "\n");

        size_t output_layer_neurons = network->num_layers;
        size_t output_layer =
            network->neural_layers[output_layer_neurons - 1].num_neurons;

        for (int i = 0; i < layer_size - 1; i++) {

                size_t num_neurons = network->neural_layers[i].num_neurons;
                fprintf(file, "Layer: %d\n", i);
                fprintf(file, "Number of Neurons: %ld\n", num_neurons);

                Layer *curr_layer = &network->neural_layers[i];
                Layer *next_layer = &network->neural_layers[i + 1];
                for (size_t j = 0; j < curr_layer->num_neurons; j++) {
                        fprintf(file, "Neuron[%zu]\t", j);
                        if (i != 0) {
                                fprintf(file, "Bias: %f\tDelta: %f\n",
                                        curr_layer->neurons[j].bias,
                                        curr_layer->neurons[j].delta);
                        }
                        for (size_t k = 0; k < next_layer->num_neurons; k++) {
                                fprintf(file, "\t\tweight[%ld][%ld]: %f\n", j,
                                        k, curr_layer->neurons[j].weight[k]);
                        }
                }

                for (int k = 0; k < 60; k++) {
                        fprintf(file, "*");
                }
                fprintf(file, "\n");
        }
        fprintf(file, "Layer: %ld\n", output_layer_neurons - 1);
        fprintf(file, "Number of Neurons: %ld\n", output_layer);

        for (size_t b = 0; b < output_layer; ++b) {
                fprintf(file, "Neuron[%zu]\tBias: %f\tDelta: %f\n", b,
                        network->neural_layers[output_layer_neurons - 1]
                            .neurons[b]
                            .bias,
                        network->neural_layers[output_layer_neurons - 1]
                            .neurons[b]
                            .delta);
        }

        fclose(file);
        printf("Model saved successfully to %s\n", file_name);
}

/* returns original network weights and biases */
NeuralNetwork *load_model(NeuralNetwork *network, const char *file_name) {

        if (!network) {
                fprintf(stderr, "Error: There's some error in network.\n");
                exit(EXIT_FAILURE);
        }

        if (!strcmp(file_name, " ") || !strcmp(file_name, "")) {
                fprintf(stderr,
                        "There is No model_file in that name %s. Please check "
                        "once !. \n",
                        file_name);
                exit(EXIT_FAILURE);
        }

        FILE *file = fopen(file_name, "r");
        if (!file) {
                fprintf(stderr, "Error while opening the file\n");
                exit(EXIT_FAILURE);
        }

        size_t total_layers_;
        fscanf(file, "Total Layers: %zu", &total_layers_);

        static char buffer[512];

        size_t *each_layer_neurons = calloc(total_layers_, sizeof(size_t));

        /* Extract each layer neurons */
        for (size_t i = 0; i < total_layers_; ++i) {
                size_t index = 0;
                fgets(buffer, sizeof(buffer), file);
                fscanf(file, "Layer[%zu]: %zu", &index, &each_layer_neurons[i]);
        }

        float **bias_ = malloc(total_layers_ * sizeof(float *));
        if (!bias_) {
                fprintf(stderr, "Error: Memory is not allocated for Bias "
                                "'Load_Model();'\n");
                assert(bias_);
        }

        float ***weight_ = malloc(total_layers_ * sizeof(float **));
        if (!weight_) {
                fprintf(stderr, "Error: Memroy is not allocated for weight "
                                "matrix 'Load_Model();'");
                assert(weight_);
        }

        for (size_t j = 0; j < total_layers_; j++) {
                bias_[j] = calloc(each_layer_neurons[j], sizeof(float));
        }

        for (size_t l = 0; l < total_layers_ - 1; l++) {
                weight_[l] = malloc(each_layer_neurons[l] * sizeof(float *));
                for (size_t j = 0; j < each_layer_neurons[l]; j++) {
                        weight_[l][j] =
                            calloc(each_layer_neurons[l + 1], sizeof(float));
                }
        }

        // extract bias and delta values from layer 1 to n - 1.
        // track layer individually
        size_t layer = 1;
        size_t neuron = 0;

        size_t l = 0;
        size_t curr_neuron = 0;
        size_t next_neuron = 0;
        while (fgets(buffer, sizeof(buffer), file)) {

                (neuron == each_layer_neurons[layer] && layer < total_layers_)
                    ? layer++,
                    neuron = 0 : neuron;

                if (l < total_layers_ - 1 &&
                    next_neuron == each_layer_neurons[l + 1]) {
                        curr_neuron++;
                        next_neuron = 0;

                        if (curr_neuron == each_layer_neurons[l]) {
                                l++;
                                curr_neuron = 0;
                        }
                }

                float temp_bias = 0.f;

                if (strstr(buffer, "Bias:") != NULL) {
                        char *bias_val = strchr(buffer, ':');
                        if (bias_val != NULL) {
                                temp_bias = atof(bias_val + 1);
                                bias_[layer][neuron] = temp_bias;
                                neuron++;
                        }
                }

                if (strstr(buffer, "weight") != NULL) {
                        char *trim_weight = strchr(buffer, ':');
                        if (trim_weight != NULL) {
                                float val = atof(trim_weight + 1);
                                weight_[l][curr_neuron][next_neuron] = val;
                        }
                        next_neuron++;
                }
        }

        printf("\n**************>Debug Bias values: \n");
        for (size_t i = 1; i < total_layers_; i++) {
                for (size_t j = 0; j < each_layer_neurons[i]; j++) {
                        network->neural_layers[i].neurons[j].bias = bias_[i][j];
                        float debug_bias =
                            network->neural_layers[i].neurons[j].bias;
                        printf("[%zu Layer][%zu Neuron]: %f\n", i, j,
                               debug_bias);
                }
                printf("\n");
        }

        printf("\n**************>Debug Weights: \n");
        for (size_t i = 0; i < total_layers_ - 1; i++) {
                Layer *CurrLayer = &network->neural_layers[i];
                for (size_t j = 0; j < each_layer_neurons[i]; j++) {
                        for (size_t k = 0; k < each_layer_neurons[i + 1]; k++) {
                                printf("Original Weights[%zu][%zu][%zu]: %f | ",
                                       i, j, k,
                                       CurrLayer->neurons[j].weight[k]);
                                CurrLayer->neurons[j].weight[k] =
                                    weight_[i][j][k];
                                printf("Weights[%zu Layer][%zu][%zu]: %f\n", i,
                                       j, k, CurrLayer->neurons[j].weight[k]);
                        }
                }
                printf("\n");
        }

        printf("**************>Debug Total Layers: %zu\n", total_layers_);
        for (size_t i = 0; i < total_layers_; i++) {
                printf("Number of Neurons present in [%zu]th Layer: %zu\n", i,
                       each_layer_neurons[i]);
        }

        fclose(file);

        /* Release memory */
        for (size_t l = 0; l < total_layers_ - 1; l++) {
                for (size_t j = 0; j < each_layer_neurons[l]; j++) {
                        free(weight_[l][j]);
                        weight_[l][j] = NULL;
                }
        }
        free(weight_);

        for (size_t j = 0; j < total_layers_; j++) {
                free(bias_[j]);
                bias_[j] = NULL;
        }
        free(bias_);

        free(each_layer_neurons);
        return network;
}

/* returns performance and accuracy of validation network */
void validate_network(NeuralNetwork *network, float **x_val, int **y_val,
                      size_t val_size, size_t input_features, _Bool debug) {

        if (!network) {
                fprintf(stderr, "Error: Something error in the network\n");
                assert(network);
        }

        if (!x_val || !y_val) {
                fprintf(stderr, "Error");
                return exit(EXIT_FAILURE);
        }
        printf("Function is in Validation phase: \n");

        printf("Validation Size: %zu | Total Samples: %zu\n", val_size,
               input_features);

        int correct_label = 0;
        size_t num_class = 3;
        for (size_t i = 0; i < val_size; i++) {
                float input_neurons_val[input_features];
                for (size_t j = 0; j < input_features; j++) {
                        input_neurons_val[j] = x_val[i][j];
                }

                forward_pass(network, input_neurons_val, debug);
                float max_val = -1e9;
                int predicted_class = -1;
                for (size_t k = 0; k < num_class; k++) {
                        float output_val =
                            network->neural_layers[network->num_layers - 1]
                                .neurons[k]
                                .val;

                        printf("Predicted value: %f\t | Original Class: %d\n",
                               output_val, y_val[i][k]);
                        if (output_val > max_val) {
                                max_val = output_val;
                                predicted_class = k;
                        }
                }

                int true_class = 0;
                for (size_t k = 0; k < num_class; k++) {
                        if (y_val[i][k] == 1) {
                                true_class = k;
                                break;
                        }
                }
                printf("True Class: %d\tPredicted Class: %d\n", true_class,
                       predicted_class);
                if (predicted_class == true_class) {
                        correct_label++;
                }
        }

        float accuracy = ((float)correct_label / val_size) * 100.0f;

        printf("Correct: %d\n", correct_label);
        printf("Accuracy: %.2f%%\n", accuracy);
        printf("val size: %zu\n", val_size);
}

void predict_(NeuralNetwork *network, float **x_test, int **y_test,
              size_t testSize, size_t input_features, size_t output_class,
              _Bool debug) {
        if (!network) {
                fprintf(stderr,
                        "Error: Network is not Initialized correctly.\n");
                return exit(EXIT_FAILURE);
        }

        if (!x_test || !y_test) {
                fprintf(stderr,
                        "Error: Look at Input_features/Output_labels.\n");
                return exit(EXIT_FAILURE);
        }

        int correct = 0;
        printf("**************> Prediction: \n");
        for (size_t i = 0; i < testSize; i++) {
                float transform_input[input_features];
                for (size_t j = 0; j < input_features; j++) {
                        transform_input[j] = x_test[i][j];
                        printf("Input Neurons: %f\n", transform_input[j]);
                }

                forward_pass(network, transform_input, debug);

                float max_val = -1e9;
                int predicted_class = -1;
                for (size_t k = 0; k < 3; k++) {
                        float output_label =
                            network->neural_layers[network->num_layers - 1]
                                .neurons[k]
                                .val;

                        printf("Original Label: %d\t|\t Predicted Label: %f\n",
                               y_test[i][k], output_label);

                        if (output_label > max_val) {
                                max_val = output_label;
                                predicted_class = k;
                        }
                }
                int true_class = 0;
                for (size_t k = 0; k < output_class; k++) {
                        if (y_test[i][k] == 1) {
                                true_class = k;
                                break;
                        }
                }
                printf("True class index[0->Setosa, 1->Versicolor, "
                       "2->Virginica]: %d\t|\tPredicted class index: %d\n",
                       true_class, predicted_class);

                printf("True label[0->Setosa, 1->Versicolor, 2->Virginica]: "
                       "%s\t|\tPredicted label: %s\n",
                       one_hot_decoder(true_class),
                       one_hot_decoder(predicted_class));

                if (predicted_class == true_class) {
                        correct++;
                }
                printf("\n");
        }

        float accuracy = ((float)correct / testSize) * 100.f;
        printf("Correct Prediction: %d / %zu\t|\tAccuracy: %f\n", correct,
               testSize, accuracy);
}
