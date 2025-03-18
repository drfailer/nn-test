#ifndef LAYER_H
#define LAYER_H
#include <cstddef>
#include "math.hpp"

struct Layer {
    Matrix weights;
    Vector biases;
    size_t nb_nodes;
    size_t nb_inputs;
};

#endif
