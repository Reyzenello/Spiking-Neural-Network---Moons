# Spiking-Neural-Network---Moons


Playing around with SNN just to see if the data are generalized well based on synthetic input: 

![image](https://github.com/Reyzenello/Spiking-Neural-Network---Moons/assets/43668563/13eecd1e-3b43-401e-a8ff-dd136ef4a85c)



This code implements a Spiking Neural Network (SNN) using PyTorch and trains it on a synthetic dataset created using `make_moons`. 

**1. Libraries and Data:**

- Imports necessary libraries, including PyTorch, Matplotlib, and Scikit-learn.
- Uses `make_moons` to generate a non-linearly separable dataset, which is good for testing the SNN's ability to learn complex patterns.
- Splits the data into training and testing sets using `train_test_split`.
- Normalizes the data using `StandardScaler`.
- Converts the data into PyTorch tensors.

**2. `SpikingNeuron` Class:**

```python
class SpikingNeuron(nn.Module):
    # ...
```

This class defines a single spiking neuron.

- `__init__`: Initializes the neuron with a `threshold` and `reset_value`. The `membrane_potential` is initialized to `None` and reset in the forward pass.
- `forward`:
    - Resets the membrane potential at the start of each forward pass.
    - Accumulates input `x` into the `membrane_potential`.
    - Generates a `spike` (1 or 0) based on whether the `membrane_potential` exceeds the `threshold`.
    - Resets the `membrane_potential` if a spike occurs.
    - Returns the `spike`.

**3. `SpikingLayer` Class:**

```python
class SpikingLayer(nn.Module):
    # ...
```

This class represents a layer of spiking neurons.

- `__init__`: Initializes the layer with a weight matrix and a `SpikingNeuron` instance.
- `forward`:
    - Computes the weighted sum of inputs using `F.linear`.
    - Passes the weighted sum to the `SpikingNeuron` to generate spikes.

**4. `SpikingNeuralNetwork` Class:**

```python
class SpikingNeuralNetwork(nn.Module):
    # ...
```

This class defines the complete SNN architecture.

- `__init__`: Creates a `ModuleList` of layers, including spiking layers for hidden layers and a linear layer for the output.
- `forward`:
    - Iterates over `num_steps` time steps.
    - Within each time step, processes the input through the spiking layers.
    - Accumulates the output from each step.
    - Calculates the mean of all spiking layer outputs and applies the final linear layer to produce the result.



**5. Training:**

- Sets parameters like `input_size`, `hidden_sizes`, `output_size`, and `num_steps`.
- Creates an instance of the `SpikingNeuralNetwork`.
- Defines the loss function (`BCEWithLogitsLoss` is suitable for binary classification) and optimizer (Adam).
- **Training loop:**
    - Iterates over epochs.
    - Clears gradients.
    - Performs a forward pass through the SNN for `num_steps` time steps.
    - Calculates the loss.
    - Computes gradients using backpropagation.
    - Updates model parameters using the optimizer.
    - Prints the loss every 10 epochs.

**6. Evaluation and Plotting:**

- Evaluates the trained model on the test set.
- Calculates accuracy.
- Plots the training data and test data with predicted labels to visualize the results.
