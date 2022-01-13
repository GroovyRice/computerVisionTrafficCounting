import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)
X = torch.rand(1, 28, 28, device=device)  # Random Tensor 28x28
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)  # Func turns Vec K -> Vec K that sums to 1
y_pred = pred_probab.argmax(1)  # Returns the index of the largest value of pred_probab
print(f"Predicted class: {y_pred}")

# Sample mini-batch of 3 images of 28x28
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten layer takes the 2D 28x28 image to a contiguous array of 784 pixel values
flatten = nn.Flatten()
flat_image = flatten(input_image)  # the mini-batch dimension (dim=0) is maintained
print(flat_image.size())

# nn.Linear applies a linear transformation on the input with the stored weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)  # Input of 28*28=784 and outputs 20
hidden1 = layer1(flat_image)  # Takes flatten image and passes it through the 1st hidden layer nn.Linear
print(hidden1.size())

# nn.ReLU is a non-linear activation that creates complex mappings between the models inputs and outputs
# nn.ReLU makes all values below zero equal to zero and anything above zero stays the same.
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential is an ordered container of modules. Data is then passed through all the modules in the order defined
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax is the final layer. Logits (values from [-inf,inf]) as passed into this layer and scaled to value [0,1]
# representing the models predicted probabilities for each class; all values sum to 1.
softmax = nn.Softmax(dim=1)  # dim is the dimension along which the values must sum to 1
pred_probab = softmax(logits)

# Can easily view each layers parameters and associated weights and biases
print("Model structure: ", model, "\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
