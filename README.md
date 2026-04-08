# neural-network-cpp

A simple neural network implementation in modern C++ for training and evaluating on the MNIST handwritten digit dataset.

## Project Overview

This project contains a lightweight neural network framework written in modern C++. It implements fundamental components including variable, layers, activation functions, and optimizers, and demonstrates how to train and test models on the MNIST dataset.

## How to Run

1. Download and process the MNIST dataset:

   ```bash
   python ./data/mnist.py
   ```

2. Generate build files:

   ```bash
   cmake -B ./build -S .
   ```

3. Enter the build directory and compile:

   ```bash
   cd ./build && make
   ```

4. Run the executable with the data directory:

   ```bash
   ./main ../data
   ```

## Results Example

```text
train image batches: 100
test image batches: 100
==================== Train ====================
[Epoch 1/20 Batch 100/100] loss = 0.0555724, accuracy = 55.0312%
[Epoch 2/20 Batch 100/100] loss = 0.0174662, accuracy = 83.2188%
[Epoch 3/20 Batch 100/100] loss = 0.0117411, accuracy = 89.2812%
[Epoch 4/20 Batch 100/100] loss = 0.00945498, accuracy = 91.5312%
[Epoch 5/20 Batch 100/100] loss = 0.00771214, accuracy = 92.875%
[Epoch 6/20 Batch 100/100] loss = 0.00640379, accuracy = 94.0625%
[Epoch 7/20 Batch 100/100] loss = 0.00530863, accuracy = 95.25%
[Epoch 8/20 Batch 100/100] loss = 0.00437111, accuracy = 96.2188%
[Epoch 9/20 Batch 100/100] loss = 0.00362152, accuracy = 96.9375%
[Epoch 10/20 Batch 100/100] loss = 0.00299147, accuracy = 97.875%
[Epoch 11/20 Batch 100/100] loss = 0.0024931, accuracy = 98.2188%
[Epoch 12/20 Batch 100/100] loss = 0.00206816, accuracy = 98.5625%
[Epoch 13/20 Batch 100/100] loss = 0.00164203, accuracy = 99.0625%
[Epoch 14/20 Batch 100/100] loss = 0.00131568, accuracy = 99.5625%
[Epoch 15/20 Batch 100/100] loss = 0.00103339, accuracy = 99.5625%
[Epoch 16/20 Batch 100/100] loss = 0.000812195, accuracy = 99.75%
[Epoch 17/20 Batch 100/100] loss = 0.000643614, accuracy = 99.9375%
[Epoch 18/20 Batch 100/100] loss = 0.000544667, accuracy = 99.9688%
[Epoch 19/20 Batch 100/100] loss = 0.000448639, accuracy = 99.9688%
[Epoch 20/20 Batch 100/100] loss = 0.000376946, accuracy = 99.9688%
==================== Test ====================
[100/100] accuracy = 90.4688%
```

## Directory Structure

- `src/`：C++ source code implementation
- `include/`：Header file declarations
- `data/`：MNIST data download and preprocessing scripts
- `build/`：CMake build output directory
