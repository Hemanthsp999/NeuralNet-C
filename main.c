#include "File/file.h"
#include "File/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {

        if (argc < 1) {
                printf("Usage: %s -dataset <name> <test-size> [-train "
                       "'epochs'|-predict]\n",
                       argv[0]);
                return 1;
        }

        char *Dataset = NULL;
        _Bool load_dataset_;
        float test_size;

        _Bool do_train, do_validate, do_predict;
        size_t epochs = 0;

        for (int i = 1; i < argc; i++) {
                if (strcmp(argv[i], "-dataset") == 0 && i + 1 < argc) {
                        Dataset = argv[++i];
                        test_size = atof(argv[++i]);
                        (Dataset) ? load_dataset_ = 1 : load_dataset_;
                }

                if (strcmp(argv[i], "-train") == 0 && i + 1 < argc) {
                        epochs = atoi(argv[++i]);
                        printf("Number of epochs: %zu\n", epochs);
                        do_train = 1;

                } else if (strcmp(argv[i], "-predict") == 0) {
                        do_predict = 1;
                } else if (strcmp(argv[i], "-val") == 0) {
                        do_validate = 1;
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

        dataset_handler *load_data = load_dataset(Dataset);

        load_data = train_test_split(load_data, test_size, 42);

        size_t in_hi_ou_neurons[] = {load_data->input_features, 10, 10, 10, 3};
        size_t *in_hi_ou_layers = in_hi_ou_neurons;

        if (do_train) {
                neural_network *init_network =
                    Feed_Forward_Network(in_hi_ou_layers, 5);

                _train_network(init_network, load_data, epochs);
        }

        if (do_validate) {
        }
        if (do_predict) {
                neural_network *init_network =
                    Feed_Forward_Network(in_hi_ou_layers, 5);

                predict_(init_network, load_data->x_test, load_data->y_test,
                         load_data->test_size, load_data->input_features);
        }

        return 0;
}
