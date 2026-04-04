import torchvision
import numpy as np

def dataset_to_numpy(dataset):
    images = []
    labels = []
    for img, label in dataset:
        img_np = np.array(img, dtype=np.float32) / 255.0
        images.append(img_np.reshape(-1))
        labels.append(label)
    images = np.array(images, dtype=np.float32)   # (N, 784)
    labels = np.array(labels, dtype=np.float32)   # (N,)
    return images, labels

def main():
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    train_images, train_labels = dataset_to_numpy(train_dataset)
    test_images,  test_labels  = dataset_to_numpy(test_dataset)

    print(f"Train images shape: {train_images.shape}")   # (60000, 784)
    print(f"Train labels shape: {train_labels.shape}")   # (60000,)
    print(f"Test images shape:  {test_images.shape}")    # (10000, 784)
    print(f"Test labels shape:  {test_labels.shape}")    # (10000,)

    train_images.tofile("./data/train_images.bin")
    train_labels.tofile("./data/train_labels.bin")
    test_images.tofile("./data/test_images.bin")
    test_labels.tofile("./data/test_labels.bin")

if __name__ == "__main__":
    main()