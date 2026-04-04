#include <cstdio>
#include <vector>
#include <iostream>

#include "../include/matrix.hpp"


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
    const size_t num_train = 60000;
    const size_t num_test  = 10000;
    const size_t pixel_dim = 784;   // 28*28

    auto train_images_arr = load_binary("./data/train_images.bin", 
                                        num_train * pixel_dim);
    auto train_labels_arr = load_binary("./data/train_labels.bin", 
                                        num_train);
    auto test_images_arr = load_binary("./data/test_images.bin", 
                                        num_test  * pixel_dim);
    auto test_labels_arr = load_binary("./data/test_labels.bin",
                                         num_test);

    auto train_images = Matrix(num_train, pixel_dim, train_images_arr);
    auto train_labels = Matrix(num_train, 1, train_labels_arr);
    auto test_images = Matrix(num_test, pixel_dim, test_images_arr);
    auto test_labels = Matrix(num_test, 1, test_labels_arr);

    std::cout << train_images.row() << " " << train_images.col() << std::endl;
    std::cout << train_labels.row() << " " << train_labels.col() << std::endl;
    std::cout << test_images.row() << " " << test_images.col() << std::endl;
    std::cout << test_labels.row() << " " << test_labels.col() << std::endl;

    std::vector<float> first_image(train_images_arr.begin(), train_images_arr.begin() + pixel_dim);
    std::cout << "\nFirst training digit (label = " << train_labels_arr[0] << "):\n";
    draw_mnist_digit(first_image, 28, 28);

    return 0;
}