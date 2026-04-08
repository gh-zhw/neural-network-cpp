#include <cstdio>
#include <vector>
#include <iostream>
#include <random>
#include <cstdint>

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

std::vector<Matrix> create_image_batches(const std::vector<float>& images,
                                         size_t batch_size,
                                         size_t pixel_dim,
                                         size_t max_batches = SIZE_MAX) {
    size_t num_samples = images.size() / pixel_dim;
    std::vector<Matrix> batches;
    for (size_t start = 0; start < num_samples && batches.size() < max_batches; start += batch_size) {
        size_t cur_batch_size = std::min(batch_size, num_samples - start);
        Matrix batch(cur_batch_size, pixel_dim);
        for (size_t i = 0; i < cur_batch_size; ++i) {
            const float* row = images.data() + (start + i) * pixel_dim;
            for (size_t j = 0; j < pixel_dim; ++j) {
                batch.set(i, j, row[j]);
            }
        }
        batches.push_back(std::move(batch));
    }
    return batches;
}

// one-hot
std::vector<Matrix> create_label_batches(const std::vector<float>& labels,
                                         size_t batch_size,
                                         size_t num_classes = 10,
                                         size_t max_batches = SIZE_MAX) {
    size_t num_samples = labels.size();
    std::vector<Matrix> batches;
    for (size_t start = 0; start < num_samples && batches.size() < max_batches; start += batch_size) {
        size_t cur_batch_size = std::min(batch_size, num_samples - start);
        Matrix batch(cur_batch_size, num_classes, 0.0f);
        for (size_t i = 0; i < cur_batch_size; ++i) {
            int cls = static_cast<int>(labels[start + i]);
            if (cls >= 0 && cls < num_classes) {
                batch.set(i, cls, 1.0f);
            }
        }
        batches.push_back(std::move(batch));
    }
    return batches;
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


int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <data_directory>" << std::endl;
        return 1;
    }

    std::string data_dir = argv[1];
    std::string train_images_path = data_dir + "/train_images.bin";
    std::string train_labels_path = data_dir + "/train_labels.bin";
    std::string test_images_path  = data_dir + "/test_images.bin";
    std::string test_labels_path  = data_dir + "/test_labels.bin";

    const size_t num_train = 60000;
    const size_t num_test  = 10000;
    const size_t pixel_dim = 784;    // 28*28
    const size_t batch_size = 32;
    const size_t num_classes = 10;
    const size_t epochs = 20;
    const size_t N = 100;              // max batches

    try {
        // load data
        auto train_images = load_binary(train_images_path.c_str(), 
                                        num_train * pixel_dim);
        auto train_labels = load_binary(train_labels_path.c_str(), 
                                        num_train);
        auto test_images = load_binary(test_images_path.c_str(), 
                                       num_test * pixel_dim);
        auto test_labels = load_binary(test_labels_path.c_str(), 
                                       num_test);

        auto train_image_batches = create_image_batches(train_images, batch_size, pixel_dim, N);
        auto train_label_batches = create_label_batches(train_labels, batch_size, num_classes, N);
        auto test_image_batches = create_image_batches(test_images, batch_size, pixel_dim, N);
        auto test_label_batches = create_label_batches(test_labels, batch_size, num_classes, N);

        std::cout << "train image batches: ";
        std::cout << train_image_batches.size() << std::endl;
        std::cout << "test image batches: ";
        std::cout << test_image_batches.size() << std::endl;

        // create model
        Linear linear_1(784, 1024, true, "linear_1");
        Linear linear_2(1024, 256, true, "linear_2");
        Linear linear_3(256, 10, true, "linear_3");

        auto params = linear_1.paramters();
        auto params_2 = linear_2.paramters();
        auto params_3 = linear_3.paramters();

        params.insert(params.end(), params_2.begin(), params_2.end());
        params.insert(params.end(), params_3.begin(), params_3.end());

        MomentumSGD optimizer(params, 0.01f, 0.9f);

        // train
        std::cout << "==================== Train ====================\n";
        for (size_t epoch = 1; epoch < epochs + 1; ++epoch) {
            size_t correct = 0;
            float epoch_loss = 0.0f;
            for (size_t i = 0; i < train_image_batches.size(); ++i) {
                std::cout << "\r[Epoch " << epoch << "/" << epochs << " Batch " << i + 1 << "/" << train_image_batches.size() << "] " << std::flush;

                auto images = train_image_batches[i];
                auto labels = train_label_batches[i];

                Variable x(images, false);
                Variable y(labels, false);

                Variable linear_1_out = linear_1.forward(x);
                Variable relu_1_out = relu(linear_1_out);
                Variable linear_2_out = linear_2.forward(relu_1_out);
                Variable relu_2_out = relu(linear_2_out);
                Variable y_pre = linear_3.forward(relu_2_out);

                Variable loss = cross_entropy_loss(y_pre, y);
                epoch_loss += loss.get(0, 0);

                for (size_t j = 0; j < y_pre.h(); ++j) {
                    float max_val = y_pre.get(j, 0);
                    size_t true_i = 0, pred_i = 0;
                    for (size_t k = 0; k < y_pre.w(); ++k) {
                        if (y.get(j, k) == 1.0f) { true_i = k; }
                        if (y_pre.get(j, k) > max_val) {
                            max_val = y_pre.get(j, k);
                            pred_i = k;
                        }
                    }
                    if (true_i == pred_i) { correct++; }
                }

                optimizer.zero_grad();
                loss.backward();
                optimizer.update();

                linear_1.clear_cache();
                linear_2.clear_cache();
                linear_3.clear_cache();
            }
            size_t num_samples = train_image_batches.size() * batch_size;
            std::cout << "loss = " << epoch_loss / num_samples << ", accuracy = " << 100.0f * correct / num_samples << "%\n";
        }

        // test
        std::cout << "==================== Test ====================\n";
        size_t correct = 0;
        for (size_t i = 0; i < test_image_batches.size(); ++i) {
                std::cout << "\r[" << i + 1 << "/" << train_image_batches.size() << "] " << std::flush;
                auto images = test_image_batches[i];
                auto labels = test_label_batches[i];

                Variable x(images, false);
                Variable y(labels, false);

                Variable linear_1_out = linear_1.forward(x);
                Variable relu_1_out = relu(linear_1_out);
                Variable linear_2_out = linear_2.forward(relu_1_out);
                Variable relu_2_out = relu(linear_2_out);
                Variable linear_3_out = linear_3.forward(relu_2_out);
                Variable y_pre = relu(linear_3_out);

                for (size_t j = 0; j < y_pre.h(); ++j) {
                    float max_val = y_pre.get(j, 0);
                    size_t true_i = 0, pred_i = 0;
                    for (size_t k = 0; k < y_pre.w(); ++k) {
                        if (y.get(j, k) == 1.0f) { true_i = k; }
                        if (y_pre.get(j, k) > max_val) {
                            max_val = y_pre.get(j, k);
                            pred_i = k;
                        }
                    }
                    if (true_i == pred_i) { correct++; }
                }

                linear_1.clear_cache();
                linear_2.clear_cache();
                linear_3.clear_cache();
            }
        size_t num_samples = train_image_batches.size() * batch_size;
        std::cout << "accuracy = " << 100.0f * correct / num_samples << "%\n";

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;

}