#include "File/file.h"
#include "File/memory.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {

        if (argc < 0) {
                printf("Usage: %s -dataset <name> <test-size> [-train "
                       "'epochs'|-predict]\n",
                       argv[0]);
                return 1;
        }

        char *Dataset = NULL;
        _Bool load_dataset_;
        float test_size;

        _Bool do_train = 0, do_validate = 0, do_predict = 0;
        _Bool debug = 0;
        size_t epochs = 0;

        for (int i = 1; i < argc; i++) {
                if (strcmp(argv[i], "-dataset") == 0 && i + 1 < argc) {
                        Dataset = argv[++i];
                        test_size = atof(argv[++i]);
                        (Dataset) ? load_dataset_ = 1 : load_dataset_;
                }

                if (strcmp(argv[i], "-train") == 0 && i + 1 < argc) {
                        epochs = atoi(argv[++i]);
                        do_train = 1;
                        (epochs == 0)
                            ? assert(epochs)
                            : printf("Number of epochs: %zu\n", epochs);
                        ;

                } else if (strcmp(argv[i], "-test") == 0) {
                        do_predict = 1;
                } else if (strcmp(argv[i], "-val") == 0) {
                        do_validate = 1;
                } else if (strcmp(argv[i], "-debug") == 0) {
                        debug = 1;
                }

                if (strcmp(argv[i], "-help") == 0) {
                        printf("Usage:\n");
                        printf("  -dataset <file> <test-size> Dataset filename "
                               "(CSV). (test-size -> 0.3)\n");
                        printf("  -train            Train the model\n");
                        printf("  -val            Validate the model\n");
                        printf(
                            "  -predict          Predict using saved model\n");
                        return 0;
                }
        }

        if (!load_dataset_)
                return fprintf(stderr, "Enter the Dataset.\n");

        if (!test_size) {
                return fprintf(stderr,
                               "Please enter a valid Train/Test size ratio\n");
        }

        int total_output_class = get_output_labels(Dataset);
        dataset_handler *load_data = load_dataset(Dataset, total_output_class);

        load_data = train_test_split(load_data, test_size, 42);
        load_data->total_output_class = total_output_class;

        size_t in_hi_ou_neurons[] = {load_data->input_features, 256,
                                     load_data->total_output_class};
        size_t *in_hi_ou_layers = in_hi_ou_neurons;

        if (do_train) {
                NeuralNetwork *init_network =
                    Feed_Forward_Network(in_hi_ou_layers, 3);

                _train_network(init_network, load_data, epochs, debug);

        } else if (do_validate) {
                NeuralNetwork *init_network =
                    Feed_Forward_Network(in_hi_ou_layers, 3);

                NeuralNetwork *load_network_weights =
                    load_model(init_network, "model_weights.txt");

                validate_network(load_network_weights, load_data->x_val,
                                 load_data->y_val, load_data->val_size,
                                 load_data->input_features, debug);
                free_network(load_network_weights);
        } else if (do_predict) {

                NeuralNetwork *init_network =
                    Feed_Forward_Network(in_hi_ou_layers, 3);
                NeuralNetwork *load_network_weights =
                    load_model(init_network, "model_weights.txt");
                predict_(load_network_weights, load_data->x_test,
                         load_data->y_test, load_data->test_size,
                         load_data->input_features,
                         load_data->total_output_class, debug);

                free_network(load_network_weights);
        }

        printf("Train ? %d\t Val ? %d\t Test %d\t Debug: %d\n", do_train,
               do_validate, do_predict, debug);
        printf("Output labels: %zu\n", load_data->total_output_class);

        free_dataset(load_data);
        return 0;
}
