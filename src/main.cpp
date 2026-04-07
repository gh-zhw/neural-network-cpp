#include <cstdio>
#include <vector>
#include <iostream>
#include <random>

#include "../include/matrix.hpp"
#include "../include/variable.hpp"
#include "../include/functions.hpp"
#include "../include/optimizer.hpp"


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

// y = x^2 + noise
std::vector<std::pair<float, float>> generate_data(int n) {
    std::vector<std::pair<float, float>> data;
    for (int i = 0; i < n; ++i) {
        float x = 2.0f * (static_cast<float>(i) / n) - 1.0f;  // [-1, 1]
        float y = x * x;
        y += 0.05f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        data.emplace_back(x, y);
    }
    return data;
}

int main() {
    try {
        const size_t N = 200;
        auto data = generate_data(N);

        // X (N,1); Y (N,1)
        Variable X(N, 1, false);
        Variable Y(N, 1, false);
        for (size_t i = 0; i < N; ++i) {
            X.set(i, 0, data[i].first);
            Y.set(i, 0, data[i].second);
        }

        Variable ones(N, 1, false);
        for (size_t i = 0; i < N; ++i) ones.set(i, 0, 1.0f);

        const size_t hidden_dim = 10;
        Variable W1(1, hidden_dim, true);   // 输入到隐藏层
        Variable b1(1, hidden_dim, true);   // 隐藏层偏置
        Variable W2(hidden_dim, 1, true);   // 隐藏层到输出
        Variable b2(1, 1, true);            // 输出层偏置

        // initialize params
        for (size_t i = 0; i < W1.h(); ++i)
            for (size_t j = 0; j < W1.w(); ++j)
                W1.set(i, j, 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f));
        for (size_t i = 0; i < b1.h(); ++i)
            for (size_t j = 0; j < b1.w(); ++j)
                b1.set(i, j, 0.0f);
        for (size_t i = 0; i < W2.h(); ++i)
            for (size_t j = 0; j < W2.w(); ++j)
                W2.set(i, j, 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f));
        b2.set(0, 0, 0.0f);

        MomentumSGD optimizer({&W1, &b1, &W2, &b2}, 0.05f, 0.9f);

        const int epochs = 500;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // hidden = relu(X * W1 + ones * b1)
            Variable XW1 = X * W1;                     // (N, hidden_dim)
            Variable b1_expanded = ones * b1;          // (N, hidden_dim)
            Variable pre_act = XW1 + b1_expanded;      // (N, hidden_dim)
            Variable hidden = relu(pre_act);           // (N, hidden_dim)

            // output = hidden * W2 + ones * b2
            Variable hiddenW2 = hidden * W2;            // (N, 1)
            Variable b2_expanded = ones * b2;           // (N, 1)
            Variable y_pred = hiddenW2 + b2_expanded;   // (N, 1)

            Variable loss = mse_loss(y_pred, Y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.update();

            if (epoch % 50 == 0) {
                float loss_val = loss.get(0, 0);
                std::cout << "Epoch " << epoch << ", Loss: " << loss_val << std::endl;
            }
        }

        std::cout << "\nFinal predictions (first 5 samples):\n";
        Variable XW1 = X * W1;
        Variable b1_expanded = ones * b1;
        Variable pre_act = XW1 + b1_expanded;
        Variable hidden = relu(pre_act);
        Variable hiddenW2 = hidden * W2;
        Variable b2_expanded = ones * b2;
        Variable y_pred = hiddenW2 + b2_expanded;
        for (size_t i = 0; i < 5; ++i) {
            float x_val = X.get(i, 0);
            float y_true = Y.get(i, 0);
            float y_pred_val = y_pred.get(i, 0);
            std::cout << "x = " << x_val << ", true = " << y_true
                    << ", pred = " << y_pred_val << std::endl;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;


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

}