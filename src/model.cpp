#include "model.hpp"
#include <random>
#include <iostream>

void Model::init() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::normal_distribution dist(.0, 1.0);

    for (auto &layer : this->layers) {
        size_t layer_weight_size = layer.nb_nodes * layer.nb_nodes_prev;
        size_t layer_biases_size = layer.nb_nodes;

        for (size_t i = 0; i < layer_weight_size; ++i) {
            layer.weights[i] = dist(gen);
        }

        for (size_t i = 0; i < layer_biases_size; ++i) {
            layer.biases[i] = dist(gen);
        }
    }
}

void Model::add_layer(size_t nb_inputs, size_t nb_nodes) {
    Layer layer;

    if (this->layers.size() > 0 && this->layers.back().nb_nodes != nb_inputs) {
        size_t lnb = this->layers.size();

        std::cerr << "error: layer " << lnb << " has "
                  << this->layers.back().nb_nodes << " nodes and layer "
                  << (lnb + 1) << " has " << nb_inputs << " inputs." <<
                  std::endl;
        clear();
        exit(1);
    }

    layer.weights = new double[nb_nodes * nb_inputs];
    layer.weights = new double[nb_nodes];
    this->layers.push_back(layer);
}

void Model::clear() {
    for (auto &layer : this->layers) {
        delete[] layer.weights;
        delete[] layer.biases;
    }
    layers.clear();
}
