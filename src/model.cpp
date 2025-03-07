#include "model.hpp"
#include <iostream>
#include <random>

void Model::init(uint64_t seed) {
    std::mt19937_64 gen(seed);
    std::normal_distribution dist(.0, 1.0);

    for (auto &layer : this->layers) {
        size_t layer_weight_size = layer.nb_nodes * layer.nb_inputs;
        size_t layer_biases_size = layer.nb_nodes;

        for (size_t i = 0; i < layer_weight_size; ++i) {
            layer.weights.mem[i] = dist(gen);
        }

        for (size_t i = 0; i < layer_biases_size; ++i) {
            layer.biases.mem[i] = dist(gen);
        }
    }
}

void Model::input(size_t nb_inputs) { this->inputs_ = nb_inputs; }

void Model::add_layer(size_t nb_nodes) {
    if (inputs_ == 0) {
        std::cerr << "error: the model must have at least 1 input. You must "
                     "configure an input layer before adding layers."
                  << std::endl;
        clear();
        exit(1);
    }
    size_t nb_inputs = inputs_;
    if (!this->layers.empty()) {
        nb_inputs = this->layers.back().nb_nodes;
    }
    this->layers.emplace_back(Matrix(nb_nodes, nb_inputs), Vector(nb_nodes),
                              nb_nodes, nb_inputs);
}

void Model::clear() { layers.clear(); }
