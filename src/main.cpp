#include <cstdio>
#include <vector>
#include <iostream>
#include <random>

#include "../include/matrix.hpp"
#include "../include/variable.hpp"
#include "../include/functions.hpp"
#include "../include/optimizer.hpp"
#include "../include/layers.hpp"


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
        Variable X(N, 1, false, "X");
        Variable Y(N, 1, false, "Y");
        for (size_t i = 0; i < N; ++i) {
            X.set(i, 0, data[i].first);
            Y.set(i, 0, data[i].second);
        }

        const size_t hidden_dim = 10;
        Linear linear1(1, hidden_dim, true, "linear_1");
        Linear linear2(hidden_dim, hidden_dim, true, "linear_2");
        Linear linear3(hidden_dim, 1, true, "linear_3");

        auto params = linear1.paramters();
        auto params2 = linear2.paramters();
        auto params3 = linear3.paramters();
        params.insert(params.end(), params2.begin(), params2.end());
        params.insert(params.end(), params3.begin(), params3.end());

        MomentumSGD optimizer(params, 0.05f, 0.9f);

        const int epochs = 500;
        for (int epoch = 1; epoch < epochs + 1; ++epoch) {
            // hidden_1 = relu(X * W1 + b1)
            Variable linear1_x = linear1.forward(X);
            linear1_x.setName("linear1_x");
            Variable hidden_1 = relu(linear1_x);
            hidden_1.setName("hidden_1");

            // hidden_2 = relu(hidden_1 * W2 + b2)
            Variable linear2_x = linear2.forward(hidden_1);
            linear2_x.setName("linear2_x");
            Variable hidden_2 = relu(linear2_x);
            hidden_2.setName("hidden_2");

            // y_pred = hidden_2 * W2 + b2
            Variable y_pred = linear3.forward(hidden_2);
            y_pred.setName("y_pred");

            Variable loss = mse_loss(y_pred, Y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.update();
            linear1.clear_cache();
            linear2.clear_cache();

            if (epoch % 50 == 0) {
                float loss_val = loss.get(0, 0);
                std::cout << "Epoch " << epoch << ", Loss: " << loss_val << std::endl;
            }
        }

        std::cout << "\nFinal predictions (first 5 samples):\n";

        Variable linear1_x = linear1.forward(X);
        Variable hidden_1 = relu(linear1_x);
        Variable linear2_x = linear2.forward(hidden_1);
        Variable hidden_2 = relu(linear2_x);
        Variable y_pred = linear3.forward(hidden_2);

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