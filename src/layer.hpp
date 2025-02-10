#ifndef LAYER_H
#define LAYER_H
#include <cstddef>

struct Layer {
    double *weights;
    double *biases;
    size_t nb_nodes;
    size_t nb_nodes_prev;
};

#endif
