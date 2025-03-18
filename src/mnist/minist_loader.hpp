#ifndef MNIST_MINIST_LOADER_H
#define MNIST_MINIST_LOADER_H
#include "../matrix.hpp"
#include "../types.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class MNISTLoader {
  public:
    using byte = char;
    using ifstream_type = std::basic_ifstream<byte>;

  public:
    unsigned int read_big_endian_uint(ifstream_type &fs) {
        byte buff[4];
        fs.read(buff, 4);
        std::swap(buff[0], buff[3]);
        std::swap(buff[1], buff[2]);
        return *reinterpret_cast<uint*>(buff);
    }

    std::vector<int> load_labels(std::string const &path) {
        ifstream_type fs(path, std::ios::binary);
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
            byte label;
            fs.read(&label, 1);
            labels[i] = int(label);
        }
        return labels;
    }

    std::vector<Vector> load_imgages(std::string const &path) {
        ifstream_type fs(path, std::ios::binary);
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
                unsigned char px_value;
                fs.get(reinterpret_cast<byte&>(px_value));
                assert(0 <= px_value && px_value <= 255);
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

    DataSet load_ds(std::string const labels_path,
                     std::string const images_path) {
        auto labels = load_labels(labels_path);
        auto images = load_imgages(images_path);

        DataSet ds(images.size());

        for (size_t i = 0; i < ds.size(); ++i) {
            ds[i] = DataSetEntry(images[i], create_output_vector(labels[i]));
        }
        return ds;
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
