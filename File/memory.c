#include "file.h"
#include <stdio.h>
#include <stdlib.h>

/* Deallocate the memory */
void free_dataset(dataset_handler *handler) {
        if (!handler)
                return;

        // Free training data
        if (handler->x_train) {
                for (size_t i = 0; i < handler->train_samples; i++) {
                        free(handler->x_train[i]);
                        handler->x_train[i] = NULL;
                }
                free(handler->x_train);
                handler->x_train = NULL;
        }

        if (handler->y_train) {
                for (size_t i = 0; i < handler->train_samples; i++) {
                        free(handler->y_train[i]); // strdup
                        handler->y_train[i] = NULL;
                }
                free(handler->y_train);
                handler->y_train = NULL;
        }

        // Free testing data
        if (handler->x_test) {
                for (size_t i = 0; i < handler->test_samples; i++) {
                        free(handler->x_test[i]);
                        handler->x_test[i] = NULL;
                }
                free(handler->x_test);
                handler->x_test = NULL;
        }

        if (handler->y_test) {
                for (size_t i = 0; i < handler->test_samples; i++) {
                        free(handler->y_test[i]); // strdup
                        handler->y_test[i] = NULL;
                }
                free(handler->y_test);
                handler->y_test = NULL;
        }

        // Free entire dataset X and Y if present
        if (handler->X) {
                for (size_t i = 0; i < handler->samples; i++) {
                        if (handler->X[i]) {
                                free(handler->X[i]);
                                handler->X[i] = NULL;
                        }
                }
                free(handler->X);
                handler->X = NULL;
        }

        if (handler->Y) {
                for (size_t i = 0; i < handler->samples; i++) {
                        if (handler->Y[i]) {
                                free(handler->Y[i]); // strdup
                                handler->Y[i] = NULL;
                        }
                }
                free(handler->Y);
                handler->Y = NULL;
        }
        free(handler->x_train);
        free(handler->y_train);
        free(handler->x_test);
        free(handler->y_test);
        free(handler->X);
        free(handler->Y);
        free(handler);
        handler = NULL;
}
