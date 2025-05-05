/* activation.c */
#include "activation.h"
#include <assert.h>
#include <math.h>

float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

float sigmoid_derivative(float x) {
        float s = sigmoid(x);
        return s * (1 - s);
}

float tan_h(float x) { return tanhf(x); }

float tanh_derivative(float x) { return 1.0 - powf(tan_h(x), 2); }

float relu(float x) { return (x > 0) ? x : 0; }

float mse(Layer *output_expected, Layer *output_predicted) {
        if (!output_expected || !output_predicted) {
                fprintf(stderr, "The given var has no contents\n");
                assert(!output_expected || !output_predicted);
        }

        size_t n = output_predicted->num_neurons;
        float sum = 0.f;

        for (size_t i = 0; i < n; i++) {
                float diff = 0.f;
                diff = output_expected->neurons[i].val -
                       output_predicted->neurons[i].val;

                sum += diff * diff;
        }

        return sum / n;
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
        }
}
