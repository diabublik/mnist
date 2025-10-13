import torch
from torchvision import datasets
import time
import numpy as np


mnist_train = datasets.MNIST(root="./data", train=True, download=False)
mnist_test = datasets.MNIST(root="./data", train=False, download=False)

images = mnist_train.data[:1000].float() / 255.0
labels = mnist_train.targets[:1000] # (1000)

one_hot_labels = torch.zeros(len(labels), 10)
one_hot_labels[torch.arange(len(labels)), labels] = 1 # (1000, 10)

test_images = mnist_test.data.float() / 255.0
test_labels = mnist_test.targets # (10 000)

one_hot_test_labels = torch.zeros(len(test_labels), 10)
one_hot_test_labels[torch.arange(len(test_labels)), test_labels] = 1 # (10 000, 10)

torch.manual_seed(1)
relu = lambda x: (x > 0) * x.float()
relu2deriv = lambda x: (x > 0)

alpha = 0.4
iterations = 350
batch_size = 8

kernel_size = 3
num_kernels = 4

conv_weights = 0.2 * torch.rand((kernel_size * kernel_size, num_kernels)) - 0.1 # (9, 4)

def extract_patches(images, kernel_size):
    batch_size, height, width = images.shape
    patches = []

    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            patch = images[:, i:i+kernel_size, j:j+kernel_size] # (batch_size, 3, 3)
            patches.append(patch.reshape(batch_size, -1)) # (676, batch_size, 9)
    
    # я не ебу как получается эта размерность, это какая то ебатория с этим dim=1
    return torch.stack(patches, dim=1) # (batch_size, 676, 9) 

hidden_size = (images.shape[1] - kernel_size + 1) * (images.shape[2] - kernel_size + 1) * num_kernels

weights_1_2 = 0.2 * torch.rand((hidden_size, 10)) - 0.1 # (2704, 10)

for j in range(iterations):
    train_error, correct_cnt = 0.0, 0.0
    start_time = time.time()

    for i in range(0, len(labels), batch_size):
        batch_start, batch_end = i, min(i + batch_size, len(labels))
        cur_batch_size = batch_end - batch_start

        layer_0 = images[batch_start:batch_end] # (batch_size, 28, 28)
        batch_labels = one_hot_labels[batch_start:batch_end] # (batch_size, 10)

        patches = extract_patches(layer_0, kernel_size) # (batch_size, 676, 9)
        conv_output = patches @ conv_weights # (batch_size, 676, 4)

        layer_1 = relu(conv_output)


        layer_1_flat = layer_1.reshape(cur_batch_size, -1) # (batch_size, 2704)

        layer_2 = layer_1_flat @ weights_1_2 # (batch_size, 10)

        train_error += torch.sum((batch_labels - layer_2) ** 2).item()
        true_labels = torch.argmax(batch_labels, dim=1)
        ind_pred_labels = torch.argmax(layer_2, dim=1) # позция наиболее вероятного правильного предикта
        correct_cnt += torch.sum(true_labels == ind_pred_labels).item()

        layer_2_delta = (batch_labels - layer_2) / cur_batch_size # (batch_size, 10)
        layer_1_delta_flat = layer_2_delta @ weights_1_2.T # (batch_size, 2704)

        layer_1_delta = layer_1_delta_flat.reshape(cur_batch_size, 676, 4) 
        layer_1_delta *= relu2deriv(layer_1) # (batch_size, 676, 4)


        conv_weights_grad = torch.zeros_like(conv_weights)
        for batch_ind in range(cur_batch_size):
            for patch_ind in range(patches.shape[1]):
                patch = patches[batch_ind:batch_ind+1, patch_ind:patch_ind+1, :] # (1, 1, 9)
                patch_grad = patch.transpose(1, 2) @ layer_1_delta[batch_ind:batch_ind+1, patch_ind:patch_ind+1, :] # (1, 9, 4)
                conv_weights_grad += patch_grad.squeeze(0) # (9, 4)
        
        weights_1_2 += alpha * (layer_1_flat.T @ layer_2_delta) 
        conv_weights += alpha * conv_weights_grad / cur_batch_size

    train_error /= len(labels)
    train_acc = correct_cnt / len(labels)

    test_error, test_correct_cnt = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(test_labels), batch_size * 4):
            batch_start, batch_end = i, min(i + batch_size * 4, len(test_labels))
            cur_batch_size = batch_end - batch_start

            layer_0 = test_images[batch_start:batch_end] # (batch_size*4, 28, 28)
            batch_test_labels = one_hot_test_labels[batch_start:batch_end]

            patches = extract_patches(layer_0, kernel_size) # (batch_size*4, 676, 9)
            conv_output = patches @ conv_weights
            layer_1 = relu(conv_output) # (batch_size*4, 676, 4)
            layer_1_flat = layer_1.reshape(layer_0.shape[0], -1) # (batch_size*4, 2704)
            
            layer_2 = layer_1_flat @ weights_1_2 # (batch_size*4, 10)

            test_error += torch.sum((batch_test_labels - layer_2) ** 2).item()
            true_test_labels = torch.argmax(layer_2, dim=1)
            ind_pred_test_labels = torch.argmax(batch_test_labels, dim=1)
            test_correct_cnt += torch.sum(ind_pred_test_labels == true_test_labels).item()
    
    test_error /= len(test_labels)
    test_acc = test_correct_cnt / len(test_labels)

    epoch_time = time.time() - start_time

    print(
        f"Epoch: {j + 1} | "
        f"Train error: {train_error:.5f} | "
        f"Train acc: {train_acc:.3f} | "
        f"Test error: {test_error:.5f} | "
        f"Test acc: {test_acc:.3f} | "
        f"Epoch time: {epoch_time:.1f}"
    )
