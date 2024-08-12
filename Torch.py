import torch
import torch.nn as nn

# Making Tensors
x = torch.tensor([1, 2, 3])
print(x)

y = torch.rand(3, 3)
print(y)

z = torch.zeros(5, 5)
print(z)


# Tensor Operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

c = a + b
print(c)

# matrix multiplication
d = torch.matmul(a.unsqueeze(0), b.unsqueeze(1))
print(d)

a.add(b)  # Adds b to a in-place
print(a)


# Autograd
x = torch.tensor(1.0, requires_grad=True)
y = 2 * x
z = y**2

# Backpropagate
z.backward()

# Gradient of z with respect to x
print(x.grad)


# Defining a nn
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Create the network
net = SimpleNN()

# Example input
input_data = torch.randn(1, 10)
output = net(input_data)
print(output)