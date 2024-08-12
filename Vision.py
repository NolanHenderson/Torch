import torch
import torchvision as torchV
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

Data_ROOT = r'C:\Users\nhenders\PycharmProjects\PyTorchStuff1\datasets'

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load the MNIST dataset
mnist_train = datasets.FashionMNIST(root=Data_ROOT, train=True, download=True, transform=transform)
mnist_test = datasets.FashionMNIST(root=Data_ROOT, train=False, download=True, transform=transform)

# Optionally, use a DataLoader
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Load a pretrained ResNet-18 model
resnet18 = models.resnet18(weights=None)

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
def imshow(img, title=None):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.show()


# Get some random training images
#dataiter = iter(train_loader)
#images, labels = next(dataiter)

# Show images
#imshow(torchV.utils.make_grid(images))


# Using a NN to classify the MNIST data set
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(28*28, 128)  # Input layer (28x28 images flattened to 784) to hidden layer
        self.fc2 = nn.Linear(128, 64)     # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(64, 10)      # Hidden layer to output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input tensor (batch_size, 28, 28) -> (batch_size, 784)
        x = F.relu(self.fc1(x))  # Apply ReLU activation function
        x = F.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.fc3(x)          # Output layer (logits)
        return x


# Create an instance of the network
model = SimpleNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer with learning rate 0.001


# Train the NN using training data
print("Training...")
num_epochs = 5  # Number of epochs to train

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')


# Test the trained model
print("Testing...")
correct = 0
total = 0

# Lists to hold images and labels
correct_images = []
correct_labels = []
incorrect_images = []
incorrect_labels = []
predicted_labels = []

# Run the model and collect data
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Compare predictions to true labels
        for i in range(len(labels)):
            if predicted[i] == labels[i]:
                correct_images.append(images[i])
                correct_labels.append(labels[i])
            else:
                incorrect_images.append(images[i])
                incorrect_labels.append(labels[i])
                predicted_labels.append(predicted[i])

        # Break after collecting enough samples
        #if len(correct_images) >= 5 and len(incorrect_images) >= 5:
        #    break

print(f'Accuracy: {100 * correct / total}%')


def show_images(images, labels, title=None):
    for i in range(len(images)):
        imshow(images[i], title=f'True label: {labels[i]}')


# Show correctly classified images
#print("Correctly classified images:")
#for i in range(5):
#    imshow(correct_images[i], title=f'True label: {correct_labels[i]}')

# Show incorrectly classified images
print("Incorrectly classified images:")
for i in range(5):
    imshow(incorrect_images[i], title=f'True label: {incorrect_labels[i]}, Predicted label: {predicted_labels[i]}')