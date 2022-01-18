import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)  # weights
b = torch.randn(3, requires_grad=True)  # biases
# requires_grad=True calculates the gradients of the loss function with respect tensors with this flag
# A tensor can be updated with x.requires_grad_(True) later if it too needs to be used to compute the gradients
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

loss.backward()  # computes the derivatives of the loss function
print(w.grad)  # tensor.grad retrieves the values from the derivative
print(b.grad)

# By default all tensors require gradient computation
z = torch.matmul(x, w)+b
print(z.requires_grad)

# torch.no_grad() prevents tensors surrounded to stop tracking computations
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Additionally tensor.detach() achieves the same result
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# Reasons for doing this includes marking parameters in nn as frozen. For fine-tuning a pretrained network
# To speed up computations when you are only doing forward pass, because computations on tensors that do not
# track gradients would be more efficient
