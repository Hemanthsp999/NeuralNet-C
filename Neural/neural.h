#ifndef Neural_
#define Neural_

typedef struct {
        float val;
        float *weight;
} Neuron;

typedef struct {
        int input_layers;
        int hidden_layers;
        int output_layers;
} Layers;

typedef struct {
        float *input_to_hidden_weights;
        float *hidden_to_output_weights;
        float *input_to_hidden_bias;
        float *hidden_to_output_bias;
} weight_bias_map;

#endif
