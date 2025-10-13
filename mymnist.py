import torch
from torchvision import datasets


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

alpha = 0.005
iterations = 350

kernel_size = 3
num_kernels = 4

conv_weights = 0.2 * torch.rand((kernel_size * kernel_size, num_kernels)) - 0.1 # (9, 4)

def extract_patches(image, kernel_size):
    batch_size, height, width = image.shape
    patches = []

    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            patch = image[:, i:i+kernel_size, j:j+kernel_size] # (1, 3, 3)
            patches.append(patch.reshape(batch_size, -1)) # (676, 9)
    
    return torch.stack(patches, dim=1) # (1, 676, 9)

hidden_size = (images.shape[1] - kernel_size + 1) * (images.shape[2] - kernel_size + 1) * num_kernels

weights_1_2 = 0.2 * torch.rand((hidden_size, 10)) - 0.1 # (2704, 10)

for j in range(iterations):
    train_error, correct_cnt = 0.0, 0.0

    for i in range(len(labels)):
        layer_0 = images[i:i+1] # (1, 28, 28)

        patches = extract_patches(layer_0, kernel_size) # (1, 676, 9)
        conv_output = patches @ conv_weights # (1, 676, 4)

        layer_1 = relu(conv_output)
        layer_1_flat = layer_1.reshape(1, -1) # (1, 2704)

        layer_2 = layer_1_flat @ weights_1_2 # (1, 10)

        train_error += torch.sum((one_hot_labels[i:i+1] - layer_2) ** 2).item()
        correct_cnt += int(labels[i] == torch.argmax(layer_2))

        layer_2_delta = one_hot_labels[i:i+1] - layer_2 # (1, 10)
        layer_1_delta_flat = layer_2_delta @ weights_1_2.T # (1, 2704)

        layer_1_delta = layer_1_delta_flat.reshape(1, 676, 4) 
        layer_1_delta *= relu2deriv(layer_1) # (1, 676, 4)

        conv_weights_grad = torch.zeros_like(conv_weights)
        for k in range(patches.shape[1]):
            patch = patches[:, k:k+1, :] # (1, 1, 9)
            patch_grad = patch.transpose(1, 2) @ layer_1_delta[:, k:k+1, :] # (1, 9, 4)
            conv_weights_grad += patch_grad.squeeze(0) # (9, 4)
        
        weights_1_2 += alpha * (layer_1_flat.T @ layer_2_delta) 
        conv_weights += alpha * conv_weights_grad

    train_error /= len(labels)
    train_acc = correct_cnt / len(labels)

    test_error, test_correct_cnt = 0.0, 0.0
    with torch.no_grad():
        for i in range(len(test_labels)):
            layer_0 = test_images[i:i+1] # (1, 28, 28)

            patches = extract_patches(layer_0, kernel_size) # (1, 676, 9)
            conv_output = patches @ conv_weights
            layer_1 = relu(conv_output) # (1, 676, 4)
            layer_1_flat = layer_1.reshape(1, -1) # (1, 2704)
            
            layer_2 = layer_1_flat @ weights_1_2 # (1, 10)

            test_error += torch.sum((one_hot_test_labels[i:i+1] - layer_2) ** 2).item()
            test_correct_cnt += int(torch.argmax(layer_2) == test_labels[i])
    
    test_error /= len(test_labels)
    test_acc = test_correct_cnt / len(test_labels)

    print(
        f"Epoch: {j + 1} | "
        f"Train error: {train_error:.5f} | "
        f"Train acc: {train_acc:.3f} | "
        f"Test error: {test_error:.5f} | "
        f"Test acc: {test_acc:.3f} | "
    )