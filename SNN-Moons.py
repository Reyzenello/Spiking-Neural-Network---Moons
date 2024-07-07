import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

class SpikingNeuron(nn.Module):
    def __init__(self, threshold=1.0, reset_value=0.0):
        super(SpikingNeuron, self).__init__()
        self.threshold = threshold
        self.reset_value = reset_value
        self.membrane_potential = None

    def forward(self, x):
        self.membrane_potential = torch.zeros_like(x)  # Reset membrane potential for each forward pass
        self.membrane_potential = self.membrane_potential + x
        spike = (self.membrane_potential >= self.threshold).float()
        self.membrane_potential = (1 - spike) * self.membrane_potential + spike * self.reset_value
        return spike

class SpikingLayer(nn.Module):
    def __init__(self, input_size, output_size, threshold=1.0, reset_value=0.0):
        super(SpikingLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.neuron = SpikingNeuron(threshold, reset_value)

    def forward(self, x):
        weighted_sum = F.linear(x, self.weight)
        return self.neuron(weighted_sum)

class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, threshold=1.0, reset_value=0.0):
        super(SpikingNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(SpikingLayer(input_size, hidden_sizes[0], threshold, reset_value))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(SpikingLayer(hidden_sizes[i-1], hidden_sizes[i], threshold, reset_value))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x, num_steps):
        outputs = []
        for _ in range(num_steps):
            layer_output = x
            for layer in self.layers[:-1]:  # Apply spiking layers
                layer_output = layer(layer_output)
            outputs.append(layer_output)
        
        # Apply the final linear layer to the mean of all time steps
        final_output = self.layers[-1](torch.stack(outputs, dim=1).mean(dim=1))
        return final_output

# Create a synthetic dataset
X, y = make_moons(n_samples=1000, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Parameters
input_size = 2
hidden_sizes = [10, 10]
output_size = 1
num_steps = 10

# Create the SNN model
snn = SpikingNeuralNetwork(input_size, hidden_sizes, output_size)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=0.01)

# Train the SNN model
num_epochs = 100

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = snn(X_train_tensor, num_steps)
    loss = criterion(outputs, y_train_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")

# Evaluate the model
with torch.no_grad():
    outputs = snn(X_test_tensor, num_steps)
    predicted = (outputs > 0).float()
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f"Accuracy: {accuracy:.4f}")

# Plot the results
plt.figure(figsize=(10, 5))

# Plot training data
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
plt.title('Training Data')

# Plot test data and predictions
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted.numpy().squeeze(), cmap='viridis')
plt.title('Test Data Predictions')

plt.show()
