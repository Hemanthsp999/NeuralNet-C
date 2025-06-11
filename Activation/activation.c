/* activation.c */
#include "activation.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

/* returns sigmoid func (scales from 0 - 1)*/
float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

float sigmoid_derivative(float x) {
        float s = sigmoid(x);
        return s * (1 - s);
}

float tan_h(float x) { return tanhf(x); }

float tanh_derivative(float x) { return 1.0 - powf(tan_h(x), 2); }

float relu(float x) { return (x > 0) ? x : 0; }

float mse(Layer *output_layer, float *output_expected) {
        if (output_layer == NULL || output_expected == NULL) {
                fprintf(stderr, "Error, The passed layer or output is empty\n");
                assert(output_expected || output_layer);
                return -1;
        }

        size_t num_classes = 3;
        float sum = 0.f;
        for (size_t i = 0; i < num_classes; i++) {
                float error = output_layer->neurons[i].val - output_expected[i];
                sum += error * error;
        }

        sum /= num_classes;
        return sum;
}

/* returns multi-class labels */
float categorical_cross_entropy(Layer *expected_output,
                                Layer *predicted_output) {
        if (!expected_output || !predicted_output) {
                fprintf(
                    stderr,
                    "The provided (expected / predicted)output is empyt. ?\n");
                assert(!expected_output || !predicted_output);
        }
        float sum = 0.f;
        float epsilon = 1e-8f;

        for (size_t i = 0; i < predicted_output->num_neurons; i++) {
                float p = predicted_output->neurons[i].val;
                float y = expected_output->neurons[i].val;
                p = fmax(p, epsilon);
                sum += -y * log(p);
        }

        return sum;
}

/* returns 0 or 1 */
float binary_cross_entropy(Layer *output_neurons_expected,
                           Layer *output_neurons_predicted) {
        if (!output_neurons_expected || !output_neurons_predicted) {
                fprintf(stderr, "The provided (expected / predicted) output "
                                "value is empyt\n");
                assert(!output_neurons_expected || !output_neurons_predicted);
        }

        /*BCE= −1/N * (∑i=1N[yilog(pi)+(1−yi)log(1−pi)]) */

        int n = output_neurons_expected->num_neurons;

        float loss = 0.f;

        for (int i = 0; i < n; i++) {
                float y = output_neurons_expected->neurons[i].val;
                float p = output_neurons_predicted->neurons[i].val;

                if (p == 0)
                        p = 1e-10;
                if (p == 1)
                        p = 1 - 1e-10;

                loss += y * log(p) + (1 - y) * log(1 - p);
        }

        loss = -loss / n;

        return loss;
}

void soft_max(Layer *output_layer_vals) {
        if (!output_layer_vals) {
                fprintf(stderr, "empyt layer.");
                assert(!output_layer_vals);
        }

        printf("Number of Neurons Passed for softmax func: %ld\n",
               output_layer_vals->num_neurons);

        /*
              softmax formula = e^zi - max(z) / sum(e^zj - max(z))
            */

        float n = output_layer_vals->num_neurons;
        float max_val = output_layer_vals->neurons[0].val;

        // check for numerical stability
        for (size_t i = 0; i < n; i++) {
                if (output_layer_vals->neurons[i].val > max_val)
                        max_val = output_layer_vals->neurons[i].val;
        }

        float denominator = 0.f;

        for (int i = 0; i < n; i++) {
                denominator +=
                    expf(output_layer_vals->neurons[i].val - max_val);
        }

        // make in-place replacement
        for (int i = 0; i < n; i++) {
                output_layer_vals->neurons[i].val =
                    expf(output_layer_vals->neurons[i].val - max_val) /
                    denominator;
                printf("Soft max value: %f\n",
                       output_layer_vals->neurons[i].val);
        }
}

void l2_regularization(neural_network *processed_network, float lambda) {
        if (processed_network == NULL) {
                fprintf(stderr,
                        "Error, The given network or output is empty.\n");
                assert(processed_network);
        }

        printf("Lambda Value: %f\n", lambda);

        float loss = 0.f;
        for (size_t i = 0; i < processed_network->num_layers - 1; i++) {
                Layer *curr_layer = &processed_network->neural_layers[i];
                Layer *next_layer = &processed_network->neural_layers[i + 1];

                for (size_t j = 0; j < curr_layer->num_neurons; j++) {
                        for (size_t k = 0; k < next_layer->num_neurons; k++) {
                                loss += curr_layer->neurons[j].val *
                                        curr_layer->neurons[j].val;
                        }
                }
        }

        loss *= lambda / 2.f;
        printf("L2 loss: %f\n", loss);
}

/* returns moment and v of adam optimizer */
void _init_adam_optimizer(Layer *layer) {
        if (!layer) {
                fprintf(stderr, "Error: The Network is corrupted.\n");
                assert(layer);
        }

        layer->m_w = malloc(layer->num_neurons * sizeof(float *));
        layer->v_w = malloc(layer->num_neurons * sizeof(float *));
        layer->m_b = calloc(layer->num_neurons, sizeof(float));
        layer->v_b = calloc(layer->num_neurons, sizeof(float));

        for (size_t i = 0; i < layer->num_neurons; i++) {
                layer->m_w[i] = calloc(layer->num_neurons, sizeof(float));
                layer->v_w[i] = calloc(layer->num_neurons, sizeof(float));
        }
}

void __adam_update(neural_network *layer, float learning_rate, int time_step) {
        if (!layer) {
                fprintf(stderr, "Error: There's some error in Layer.\n");
                assert(layer);
                return exit(EXIT_FAILURE);
        }

        printf("The Learning Rate for Optimizer: %f\t Time_Step: %d\n",
               learning_rate, time_step);

        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float epsilon = 1e-8f;

        for (size_t i = 0; i < layer->num_layers; i++) {
                Layer *curr_layer = &layer->neural_layers[i];
                Layer *next_layer = &layer->neural_layers[i + 1];
                for (size_t j = 0; j < curr_layer->num_neurons; j++) {
                        for (size_t k = 0; k < next_layer->num_neurons; k++) {
                                float weight_gradient =
                                    get_weight_gradient(layer, i, j, k);

                                // update momentum (first momentum)
                                curr_layer->m_w[i][j] =
                                    beta1 * curr_layer->m_w[i][j] +
                                    (1.0f - beta1) * weight_gradient;

                                // update velocity (second momentum)
                                curr_layer->v_w[i][j] =
                                    beta2 * curr_layer->v_w[i][j] +
                                    (1.0f - beta2) * weight_gradient *
                                        weight_gradient;

                                float m_hat = curr_layer->m_w[i][j] /
                                              (1.0f - powf(beta1, time_step));
                                float v_hat = curr_layer->v_w[i][j] /
                                              (1.0f - powf(beta2, time_step));

                                // update weight
                                curr_layer->neurons[j].weight[k] -=
                                    learning_rate * m_hat /
                                    (sqrtf(v_hat) + epsilon);
                        }
                }
        }
}

float get_weight_gradient(neural_network *network, size_t layer_i,
                          size_t neuron_j, size_t neuron_k) {
        return network->neural_layers[layer_i].neurons[neuron_j].deltas *
               network->neural_layers[layer_i + 1].neurons[neuron_k].val;
}
