import torch
import numpy as np

# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another Tensor
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

shape = (2, 3,)  # 2 rows, 3 columns
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

tensor = torch.rand(3, 4)  # 3 rows, 4 columns
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Moves Tensor from CPU to GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: {tensor.device}")  # Device will return cuda:0 if true

tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])
tensor[:, 1] = 73  # Makes Column 1 all 73
print(tensor)
tensor[2:, 2:] = 0  # From column 2->end and row 2->end all zeros
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)  # Concatenates along columns meaning shape(4,4) is now shape(4,12)
print(t1)
t2 = torch.cat([tensor, tensor, tensor], dim=0)  # Concatenates along rows meaning shape(4,4) is now shape(12,4)
print(t2)

y1 = tensor @ tensor.T  # tensor.T is the transpose of tensor. The @ multiples tensor A by B -> (A @ B == A * B)
y2 = tensor.matmul(tensor.T)  # tensor is "matrix multiplied" by tensor.T (the transpose)

y3 = torch.rand_like(tensor)  # y3 becomes a tensor with same shape as "tensor"
torch.matmul(tensor, tensor.T, out=y3)  # Matrix multiply tensor * tensor.T (transpose) and save at out y3
# All 3 y1, y2 and y3 will equal the same value
print(y1, y2, y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1, z2, z3)

agg = tensor.sum()  # Makes a Single Element Tensor
print(agg, type(agg))
agg_item = agg.item()  # Converts to a single item
print(agg_item, type(agg_item))

tensor = torch.ones(4,3)
tensor[:, 1] = 0
print(tensor, "\n")
tensor.add_(5)  # In-place operation, seems like an element-wise operation but saves to tensor object
print(tensor)
a1 = tensor.t_()  # Both a1 and tensor will be the same due to the in-place operation
print(a1, "\n", tensor)
a2 = tensor.t()  # this doesn't update the tensor variable thus a2 and tensor are different
print(a2, "\n", tensor)
# In-place operations are like pretty much just this:
tensor = torch.t(tensor)  # which is the same as tensor.t_()

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()  # Converts to numpy array
print(f"n: {n}")

t.add_(1)  # In-place operations also update the numpy array
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)  # Similarly, changes to NumPy arrays change the tensor
print(f"t: {t}")
print(f"n: {n}")

n = np.zeros(5)  # Updating the value doesn't reflect on tensor and vice-versa
print(f"t: {t}")
print(f"n: {n}")
