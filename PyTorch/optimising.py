import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


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


model = NeuralNetwork()
learning_rate = 1e-3  # How much to update model parameters at each batch/epoch
batch_size = 64  # the number of data samples propagated through the network before the parameters are updated
epochs = 10  # number of times to iterate over the dataset

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()  # normalises and computes the prediction error
# Other Loss Func include: nn.MSELoss (Mean Square Error) for regression tasks, and nn.NLLLoss (Negative Log Likelihood)
# for classification. nn.CrossEntropyLoss uses both nn.LogSoftmax and nn.NLLLoss

optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent
# This registers the models parameters and the learning rate needed (the optimizer adjusts the models parameters to
# reduce model error between each training step)
# The optimiser works in 3 steps:
# 1) reset the gradients of the models parameters, zero them at each iteration (optimizer.zero_grad()).
# 2) Back-propagate the prediction loss (loss.backward()). Each param has its own gradient.
# 3) optimizer.step() adjusts the parameters by the gradients collected in the backward pass.


def train_loop(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimiser)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
