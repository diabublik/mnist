import torch
import numpy as np
from torchvision import datasets
import sys

mnist_train = datasets.MNIST(root="./data", train=True, download=True)
mnist_test = datasets.MNIST(root="./data", train=False, download=True)

images = mnist_train.data[:1000].float().reshape(1000, 28*28) / 255.0
labels = mnist_train.targets[:1000]

one_hot_labels = torch.zeros(len(labels), 10)
one_hot_labels[torch.arange(len(labels)), labels] = 1

test_images = mnist_test.data.float().reshape(10000, 28*28) / 255.0
test_labels = mnist_test.targets

one_hot_test_labels = torch.zeros(len(test_labels), 10)
one_hot_test_labels[torch.arange(len(test_labels)), test_labels] = 1

torch.manual_seed(1)

relu = lambda x: x.float() * (x > 0)
relu2deriv = lambda x: (x > 0)

alpha, iterations, hidden_size, pixels_in_image, num_labels = 0.005, 350, 40, 784, 10

weights_0_1 = 0.2 * torch.rand((pixels_in_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * torch.rand((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    error, correct_count = 0.0, 0.0
    
    for i in range(len(labels)):
        layer_0 = images[i:i+1]
        layer_1 = relu(layer_0 @ weights_0_1)
        layer_2 = layer_1 @ weights_1_2

        error += torch.sum((one_hot_labels[i:i+1] - layer_2) ** 2).item()
        correct_count += int(torch.argmax(layer_2) == labels[i])

        layer_2_delta = one_hot_labels[i:i+1] - layer_2
        layer_1_delta = layer_2_delta @ weights_1_2.T * relu2deriv(layer_1)

        weights_1_2 += alpha * (layer_1.T * layer_2_delta)
        weights_0_1 += alpha * (layer_0.T * layer_1_delta)

    train_error = error / len(labels)
    train_acc = correct_count / len(labels)

    test_error, test_cor_count = 0.0, 0.0

    for i in range(len(test_labels)):
        test_layer_0 = test_images[i:i+1]
        test_layer_1 = relu(test_layer_0 @ weights_0_1)
        test_layer_2 = test_layer_1 @ weights_1_2

        test_error += torch.sum((one_hot_test_labels[i:i+1] - test_layer_2) ** 2).item()
        test_cor_count += int(torch.argmax(test_layer_2) == test_labels[i])

    test_error /= len(test_labels)
    test_acc = test_cor_count / len(test_labels)

    print(f"Epoch = {j + 1} | train_acc = {train_acc} | test_acc = {test_acc}")
