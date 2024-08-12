import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

Data_ROOT = r'C:\Users\nhenders\PycharmProjects\PyTorchStuff1\datasets'

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load the MNIST dataset
mnist_train = datasets.MNIST(root=Data_ROOT, train=True, download=False, transform=transform)
mnist_test = datasets.MNIST(root=Data_ROOT, train=False, download=False, transform=transform)

# Optionally, use a DataLoader
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Load a pretrained ResNet-18 model
resnet18 = models.resnet18(pretrained=True)

# Switch the model to evaluation mode
resnet18.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),          # Resize image to 128x128
    transforms.RandomHorizontalFlip(),      # Randomly flip the image horizontally
    transforms.ToTensor(),                  # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))    # Normalize the image
])

# Applying the transformation to an image
image = datasets.MNIST(root=Data_ROOT, transform=transform)


# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # Un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))