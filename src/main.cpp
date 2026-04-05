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
        std::cout << "========== Test 1: Basic Arithmetic ==========\n";
        Variable a(2, 2, true);
        Variable b(2, 2, true);
        a.set(0,0,1); a.set(0,1,2); a.set(1,0,3); a.set(1,1,4);
        b.set(0,0,5); b.set(0,1,6); b.set(1,0,7); b.set(1,1,8);

        std::cout << "a:\n"; a.printValue();
        std::cout << "b:\n"; b.printValue();

        Variable c = a + b;
        std::cout << "c = a + b:\n"; c.printValue();
        c.backward();
        std::cout << "After backward, a.grad (should be all 1):\n"; a.printGrad();
        std::cout << "b.grad (should be all 1):\n"; b.printGrad();

        // Reset gradients (假设有 zero_grad 方法)
        a.zero_grad(); b.zero_grad();
        Variable d = a - b;
        std::cout << "\nd = a - b:\n"; d.printValue();
        d.backward();
        std::cout << "a.grad (should be all 1):\n"; a.printGrad();
        std::cout << "b.grad (should be all -1):\n"; b.printGrad();

        a.zero_grad(); b.zero_grad();
        Variable e = a.elementMul(b);   // 逐元素乘法
        std::cout << "\ne = a * b (element-wise):\n"; e.printValue();
        e.backward();
        std::cout << "a.grad (should equal b):\n"; a.printGrad();
        std::cout << "b.grad (should equal a):\n"; b.printGrad();

        // 矩阵乘法
        Variable A(2, 3, true);
        Variable B(3, 2, true);
        for (size_t i=0; i<2; ++i)
            for (size_t j=0; j<3; ++j)
                A.set(i,j, i*3+j+1);
        for (size_t i=0; i<3; ++i)
            for (size_t j=0; j<2; ++j)
                B.set(i,j, i*2+j+1);
        std::cout << "\nA (2x3):\n"; A.printValue();
        std::cout << "B (3x2):\n"; B.printValue();
        Variable C = A * B;
        std::cout << "C = A * B:\n"; C.printValue();
        C.backward();
        std::cout << "A.grad (should be C.grad * B^T):\n"; A.printGrad();
        std::cout << "B.grad (should be A^T * C.grad):\n"; B.printGrad();

        std::cout << "\n========== Test 2: Activation Functions ==========\n";
        Variable x(2, 2, true);
        x.set(0,0, -1); x.set(0,1, 0); x.set(1,0, 2); x.set(1,1, 3);
        std::cout << "x:\n"; x.printValue();

        Variable r = relu(x);
        std::cout << "relu(x):\n"; r.printValue();
        r.backward();
        std::cout << "Gradient w.r.t x (should be 1 where x>0 else 0):\n"; x.printGrad();

        x.zero_grad();
        Variable s = sigmoid(x);
        std::cout << "sigmoid(x):\n"; s.printValue();
        s.backward();
        std::cout << "Gradient w.r.t x (sigmoid'(x) = s*(1-s)):\n"; x.printGrad();

        std::cout << "\n========== Test 3: Loss Functions ==========\n";
        // 交叉熵损失: 2个样本，3个类别
        Variable logits(2, 3, true);
        logits.set(0,0, 2.0); logits.set(0,1, 1.0); logits.set(0,2, 0.5);
        logits.set(1,0, 0.5); logits.set(1,1, 2.0); logits.set(1,2, 1.0);
        Variable labels(2, 3, false);
        labels.set(0,0, 1.0);   // 样本0类别0
        labels.set(1,1, 1.0);   // 样本1类别1

        std::cout << "Logits:\n"; logits.printValue();
        std::cout << "One-hot labels:\n"; labels.printValue();

        Variable ce_loss = cross_entropy_loss(logits, labels);
        std::cout << "Cross entropy loss:\n"; ce_loss.printValue();
        ce_loss.backward();
        std::cout << "Gradient w.r.t logits (should be (softmax - labels)/batch_size):\n";
        logits.printGrad();

        // MSE损失
        Variable pred(2, 2, true);
        Variable target(2, 2, false);
        pred.set(0,0, 1.5); pred.set(0,1, 2.5);
        pred.set(1,0, 3.5); pred.set(1,1, 4.5);
        target.set(0,0, 1.0); target.set(0,1, 2.0);
        target.set(1,0, 3.0); target.set(1,1, 4.0);
        std::cout << "\nPredictions:\n"; pred.printValue();
        std::cout << "Targets:\n"; target.printValue();

        Variable mse = mse_loss(pred, target);
        std::cout << "MSE loss:\n"; mse.printValue();
        mse.backward();
        std::cout << "Gradient w.r.t predictions (should be 2*(pred-target)/(N*D)):\n";
        pred.printGrad();

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}