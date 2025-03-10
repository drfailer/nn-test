#ifndef MNIST_MINIST_LOADER_H
#define MNIST_MINIST_LOADER_H
#include "../matrix.hpp"
#include "../types.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class MNISTLoader {
  public:
    unsigned int read_big_endian_uint(std::ifstream &fs) {
        char buff[4];
        fs.read(buff, 4);
        std::swap(buff[0], buff[3]);
        std::swap(buff[1], buff[2]);
        return *reinterpret_cast<uint*>(buff);
    }

    std::vector<int> load_labels(std::string const &path) {
        std::ifstream fs(path);
        [[maybe_unused]] unsigned int magic = 0, size = 0;

        if (!fs.is_open()) {
            std::cerr << "error: can't open label file " << path << std::endl;
            return {};
        }
        std::cout << "loading labels " << path << "..." << std::endl;

        magic = read_big_endian_uint(fs);
        size = read_big_endian_uint(fs);

        std::cout << "magic = " << magic << "; size = " << size << std::endl;

        std::vector<int> labels(size);
        for (size_t i = 0; i < size; ++i) {
            char label;
            fs.read(&label, 1);
            labels[i] = label;
        }
        return labels;
    }

    std::vector<Vector> load_imgages(std::string const &path) {
        std::ifstream fs(path);
        [[maybe_unused]] unsigned int magic = 0, size = 0, rows = 0, cols = 0;

        if (!fs.is_open()) {
            std::cerr << "error: can't open image file " << path << std::endl;
            return {};
        }
        std::cout << "loading images " << path << "..." << std::endl;

        magic = read_big_endian_uint(fs);
        size = read_big_endian_uint(fs);
        rows = read_big_endian_uint(fs);
        cols = read_big_endian_uint(fs);

        std::cout << "magic = " << magic << "; size = " << size << std::endl;
        std::cout << "row & cols = " << rows << "x" << cols << std::endl;

        std::vector<Vector> images(size);

        for (size_t i = 0; i < size; ++i) {
            Vector image(rows * cols);
            for (size_t px = 0; px < image.size; ++px) {
                char px_value;
                fs.read(&px_value, 1);
                image[px] = (double) px_value / 255.;
            }
            images[i] = std::move(image);
        }

        return images;
    }

    Vector create_output_vector(int label) {
        Vector result(10);
        memset(result.mem, 0, result.size * sizeof(*result.mem));
        result[label] = 1;
        return result;
    }

    DataBase load_db(std::string const labels_path,
                     std::string const images_path) {
        auto labels = load_labels(labels_path);
        auto images = load_imgages(images_path);

        DataBase db(images.size());

        for (size_t i = 0; i < db.size(); ++i) {
            db[i] = DataBaseEntry(images[i], create_output_vector(labels[i]));
        }
        return db;
    }

    static void print_image(Vector const &image, size_t rows, size_t cols) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                if (image[i * cols + j] == 0) {
                    std::cout << "  ";
                } else {
                    std::cout << "##";
                }
            }
            std::cout << std::endl;
        }
    }
};

#endif
