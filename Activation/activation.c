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

void l2_regularization(NeuralNetwork *processed_network, float lambda) {
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

/* Initialize Adam optimizer arrays */
void _init_adam_optimizer(NeuralNetwork *network) {
        // Initialize Adam optimizer arrays
        if (!network) {
                fprintf(stderr,
                        "Error: Network is NULL in init_adam_optimizer\n");
                return;
        }

        printf("Initializing Adam optimizer for %zu layers\n",
               network->num_layers);

        for (size_t i = 0; i < network->num_layers; i++) {
                Layer *layer = &network->neural_layers[i];

                // Initialize bias momentum arrays for all layers except input
                // (if input has no trainable bias)
                layer->m_b = (float *)calloc(layer->num_neurons, sizeof(float));
                layer->v_b = (float *)calloc(layer->num_neurons, sizeof(float));

                if (!layer->m_b || !layer->v_b) {
                        fprintf(stderr,
                                "Error: Failed to allocate bias momentum "
                                "arrays for layer %zu\n",
                                i);
                        exit(EXIT_FAILURE);
                }

                // Initialize weight momentum arrays (not needed for output
                // layer)
                if (i < network->num_layers - 1) {
                        size_t next_layer_size =
                            network->neural_layers[i + 1].num_neurons;

                        layer->m_w = (float **)malloc(layer->num_neurons *
                                                      sizeof(float *));
                        layer->v_w = (float **)malloc(layer->num_neurons *
                                                      sizeof(float *));

                        if (!layer->m_w || !layer->v_w) {
                                fprintf(stderr,
                                        "Error: Failed to allocate weight "
                                        "momentum arrays for layer %zu\n",
                                        i);
                                exit(EXIT_FAILURE);
                        }

                        for (size_t j = 0; j < layer->num_neurons; j++) {
                                layer->m_w[j] = (float *)calloc(next_layer_size,
                                                                sizeof(float));
                                layer->v_w[j] = (float *)calloc(next_layer_size,
                                                                sizeof(float));

                                if (!layer->m_w[j] || !layer->v_w[j]) {
                                        fprintf(stderr,
                                                "Error: Failed to allocate "
                                                "momentum arrays for neuron "
                                                "%zu in layer %zu\n",
                                                j, i);
                                        exit(EXIT_FAILURE);
                                }
                        }
                } else {
                        // Output layer doesn't need weight momentum arrays
                        layer->m_w = NULL;
                        layer->v_w = NULL;
                }
        }

        printf("Adam optimizer initialization completed\n");
}

/* returns moment and velocity of adam optimizer */
void __adam_update(NeuralNetwork *network, float learning_rate, int time_step) {
        if (!network) {
                fprintf(stderr, "Error: Network is NULL.\n");
                exit(EXIT_FAILURE);
        }

        printf("The Learning Rate for Optimizer: %f\t Time_Step: %d\n",
               learning_rate, time_step);

        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float epsilon = 1e-8f;

        // Update weights between layers
        for (size_t i = 0; i < network->num_layers - 1; i++) {
                Layer *current_layer = &network->neural_layers[i];
                Layer *next_layer = &network->neural_layers[i + 1];
                printf("Processing Epoch in adam_optimizer: %ld\n", i + 1);

                // Check if momentum arrays are allocated
                if (!current_layer->m_w || !current_layer->v_w) {
                        fprintf(stderr,
                                "Error: Momentum arrays not initialized for "
                                "layer %zu\n",
                                i);
                        assert(current_layer->m_w || current_layer->v_w);
                }

                for (size_t j = 0; j < current_layer->num_neurons; j++) {
                        // Check if weight array exists
                        if (!current_layer->neurons[j].weight) {
                                fprintf(stderr,
                                        "Error: Weight array not allocated for "
                                        "neuron %zu in layer %zu\n",
                                        j, i);
                                continue;
                        }

                        for (size_t k = 0; k < next_layer->num_neurons; k++) {
                                // Calculate weight gradient directly here
                                // Gradient = input_activation * output_error

                                float weight_gradient =
                                    get_weight_gradient(network, i, j, k);

                                // Update first moment (momentum)
                                current_layer->m_w[j][k] =
                                    beta1 * current_layer->m_w[j][k] +
                                    (1.0f - beta1) * weight_gradient;
                                // Update second moment (velocity)
                                current_layer->v_w[j][k] =
                                    beta2 * current_layer->v_w[j][k] +
                                    (1.0f - beta2) * weight_gradient *
                                        weight_gradient;

                                // Bias correction
                                float m_hat = current_layer->m_w[j][k] /
                                              (1.0f - powf(beta1, time_step));
                                float v_hat = current_layer->v_w[j][k] /
                                              (1.0f - powf(beta2, time_step));

                                // Update weight
                                current_layer->neurons[j].weight[k] -=
                                    learning_rate * m_hat /
                                    (sqrtf(v_hat) + epsilon);
                        }
                }
        }

        // Update biases (skip input layer - layer 0)
        for (size_t i = 1; i < network->num_layers; i++) {
                Layer *layer = &network->neural_layers[i];

                // Check if bias momentum arrays are allocated
                if (!layer->m_b || !layer->v_b) {
                        fprintf(stderr,
                                "Error: Bias momentum arrays not initialized "
                                "for layer %zu\n",
                                i);
                        continue;
                }

                for (size_t j = 0; j < layer->num_neurons; j++) {
                        // Bias gradient is just the error delta
                        float bias_gradient = layer->neurons[j].delta;

                        // Update first moment (momentum) for bias
                        layer->m_b[j] = beta1 * layer->m_b[j] +
                                        (1.0f - beta1) * bias_gradient;

                        // Update second moment (velocity) for bias
                        layer->v_b[j] =
                            beta2 * layer->v_b[j] +
                            (1.0f - beta2) * bias_gradient * bias_gradient;

                        // Bias correction
                        float m_hat_bias =
                            layer->m_b[j] / (1.0f - powf(beta1, time_step));
                        float v_hat_bias =
                            layer->v_b[j] / (1.0f - powf(beta2, time_step));

                        // Update bias
                        layer->neurons[j].bias -= learning_rate * m_hat_bias /
                                                  (sqrtf(v_hat_bias) + epsilon);
                }
        }
}

/* returns weight gradient difference of curr and next layer*/
float get_weight_gradient(NeuralNetwork *network, size_t layer_i,
                          size_t neuron_j, size_t neuron_k) {
        return (network->neural_layers[layer_i].neurons[neuron_j].val *
                network->neural_layers[layer_i + 1].neurons[neuron_k].delta);
}
