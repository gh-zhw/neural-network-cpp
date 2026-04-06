#include <cstdio>
#include <vector>
#include <iostream>

#include "../include/matrix.hpp"
#include "../include/variable.hpp"
#include "../include/functions.hpp"


std::vector<float> load_binary(const char* filename, size_t expected_elements) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t required_size = expected_elements * sizeof(float);
    if (file_size != required_size) {
        std::cerr << "File size mismatch: expected " << required_size
                  << " bytes, got " << file_size << " bytes" << std::endl;
        fclose(f);
        return {};
    }

    std::vector<float> data(expected_elements);
    fread(data.data(), sizeof(float), expected_elements, f);
    fclose(f);
    return data;
}

void draw_mnist_digit(const std::vector<float>& pixels, int width, int height) {
    if (pixels.size() != width * height) {
        std::cerr << "Pixel count mismatch: expected " << width * height
                  << ", got " << pixels.size() << std::endl;
        return;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float v = pixels[y * width + x];
            int gray = 232 + static_cast<int>(v * 23.0f);
            std::cout << "\033[48;5;" << gray << "m  \033[0m";
        }
        std::cout << "\n"; 
    }
}

void test_long_computation_graph() {
    std::cout << "\n========== Test: Long Computation Graph ==========\n";
    
    // Input X (2x3)
    Variable X(2, 3, true);
    X.set(0,0,1); X.set(0,1,2); X.set(0,2,3);
    X.set(1,0,4); X.set(1,1,5); X.set(1,2,6);
    
    // Weight W1 (3x4)
    Variable W1(3, 4, true);
    for (size_t i=0; i<3; ++i)
        for (size_t j=0; j<4; ++j)
            W1.set(i,j, static_cast<float>(i*4+j+1) / 10.0f);  // small random
    
    // First linear: Z1 = X * W1 (2x4)
    Variable Z1 = X * W1;
    
    // ReLU activation
    Variable A1 = relu(Z1);
    
    // Weight W2 (4x2)
    Variable W2(4, 2, true);
    for (size_t i=0; i<4; ++i)
        for (size_t j=0; j<2; ++j)
            W2.set(i,j, static_cast<float>(i*2+j+1) / 10.0f);
    
    // Second linear: Z2 = A1 * W2 (2x2)
    Variable Z2 = A1 * W2;
    
    // Sigmoid activation
    Variable Y_pred = sigmoid(Z2);
    
    // Target matrix (2x2)
    Variable Y_true(2, 2, false);
    Y_true.set(0,0,0.2); Y_true.set(0,1,0.7);
    Y_true.set(1,0,0.9); Y_true.set(1,1,0.4);
    
    // MSE loss
    Variable loss = mse_loss(Y_pred, Y_true);
    
    std::cout << "Predicted Y:\n"; Y_pred.printValue();
    std::cout << "Loss value: "; loss.printValue();
    
    // Backward
    loss.backward();
    
    std::cout << "\nGradients:\n";
    std::cout << "X.grad:\n"; X.printGrad();
    std::cout << "W1.grad:\n"; W1.printGrad();
    std::cout << "W2.grad:\n"; W2.printGrad();
    std::cout << "Z1.grad:\n"; Z1.printGrad();
    std::cout << "A1.grad:\n"; A1.printGrad();
    std::cout << "Z2.grad:\n"; Z2.printGrad();
    std::cout << "Y_pred.grad:\n"; Y_pred.printGrad();
}


int main() {
    // const size_t num_train = 60000;
    // const size_t num_test  = 10000;
    // const size_t pixel_dim = 784;   // 28*28

    // auto train_images_arr = load_binary("./data/train_images.bin", 
    //                                     num_train * pixel_dim);
    // auto train_labels_arr = load_binary("./data/train_labels.bin", 
    //                                     num_train);
    // auto test_images_arr = load_binary("./data/test_images.bin", 
    //                                     num_test  * pixel_dim);
    // auto test_labels_arr = load_binary("./data/test_labels.bin",
    //                                      num_test);

    // auto train_images = Matrix(num_train, pixel_dim, train_images_arr);
    // auto train_labels = Matrix(num_train, 1, train_labels_arr);
    // auto test_images = Matrix(num_test, pixel_dim, test_images_arr);
    // auto test_labels = Matrix(num_test, 1, test_labels_arr);

    // std::cout << train_images.row() << " " << train_images.col() << std::endl;
    // std::cout << train_labels.row() << " " << train_labels.col() << std::endl;
    // std::cout << test_images.row() << " " << test_images.col() << std::endl;
    // std::cout << test_labels.row() << " " << test_labels.col() << std::endl;

    // std::vector<float> first_image(train_images_arr.begin(), train_images_arr.begin() + pixel_dim);
    // std::cout << "\nFirst training digit (label = " << train_labels_arr[0] << "):\n";
    // draw_mnist_digit(first_image, 28, 28);

    try {
        test_long_computation_graph();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}