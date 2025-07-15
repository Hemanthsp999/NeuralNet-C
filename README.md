<h1>
  <img src="images/neural_image.svg" width="100" height="100" style="vertical-align:middle;"/> NeuralNet-C
</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project implements a fully connected feedforward neural network from scratch in pure C. It includes the entire training lifecycle, custom memory management,
dataset parsing, and support for CLI-driven training, validation, and testing — all without external ML or linear algebra libraries.
The architecture closely resembles how modern deep learning frameworks operate internally, but is intentionally low-level and optimized for educational transparency,
systems control, and deployment in constrained environments.

## How to use ?

The model can be used in two ways: 
     1. Using command flags like -trian, -val and -test | 2. Use it as library

Clone the repo and cd to project_dir

```bash
git clone <project_repo>
cd "project_repo"
```

Make sure the system has zig installed. If not, visit official website and install.
After zig installation, run:

```bash
zig build
```

finally its ready to use.

## Using command flags:

1. Train network

```bash
./zig-out/bin/test -dataset "dataset_name.csv" "train-test-split ratio" -train "epochs"
```

2. Validate Network

```bash
./zig-out/bin/test -dataset "dataset_name.csv" "train-test-split ratio" -val
```

3. Test the network

```bash
./zig-out/bin/test -dataset "dataset_name.csv" "train-test-split ratio" -test
```

4. Help(if required)

```bash
./zig-out/bin/test -help
```

## Library:

1. Load and split dataset. Assign neurons and size of layers.

```bash
                const* dataset_name = "dataset_name";

                /* Get number of output labels that model has */
                int total_output_class = get_output_labels(dataset_name);

                /* load dataset */
                dataset_handler* load_data = load_dataset(dataset_name);

                /* split the dataset into train, val and test */
                load_data = train_test_split(load_data, test_size, 42); // (dataset, train/test ratio, random_state)

                /* assign number of neurons you want in each layer excluding input and output layer */
                size_t in_hi_ou_neurons[] = {load_data->input_features, 256, load_data->total_output_class}; // {load_data->input_features, 10, 10, load_data->total_output_class}
                size_t *in_hi_ou_layers = in_hi_ou_neurons;

```

2. Initialize, Train network and save model weights.

```bash
                _Bool debug = 0; // If you want to debug ? change it to 1

                /* Initialize network weights and biases*/
                NeuralNetwork *init_network =
                    Feed_Forward_Network(in_hi_ou_layers, 3);

                /* Train Network */
                _train_network(init_network, load_data, epochs, debug);
```

After training it generate a "model_weights.txt" file.

3. Load weights and Val/Test the network

Validation:

```bash
                NeuralNetwork *init_network =
                    Feed_Forward_Network(in_hi_ou_layers, 3);

                /* load model weights */
                NeuralNetwork *load_network_weights =
                    load_model(init_network, "model_weights.txt");

                validate_network(load_network_weights, load_data->x_val,
                                 load_data->y_val, load_data->val_size,
                                 load_data->input_features, debug);
```

Testing:

```bash
                NeuralNetwork *init_network =
                    Feed_Forward_Network(in_hi_ou_layers, 3);

                /* load model weights */
                NeuralNetwork *load_network_weights =
                    load_model(init_network, "model_weights.txt");

                predict_(load_network_weights, load_data->x_test,
                         load_data->y_test, load_data->test_size,
                         load_data->input_features,
                         load_data->total_output_class, debug);
```

4. Most importantly **deallocate** the memory.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
