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

void Model::add_layer(size_t nb_inputs, size_t nb_nodes) {
    if (this->layers.size() > 0 && this->layers.back().nb_nodes != nb_inputs) {
        size_t lnb = this->layers.size();

        std::cerr << "error: layer " << lnb << " has "
                  << this->layers.back().nb_nodes << " nodes and layer "
                  << (lnb + 1) << " has " << nb_inputs << " inputs."
                  << std::endl;
        clear();
        exit(1);
    }

    this->layers.emplace_back(Matrix(nb_nodes, nb_inputs), Vector(nb_nodes),
                              nb_nodes, nb_inputs);
}

void Model::clear() { layers.clear(); }
