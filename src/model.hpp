#ifndef MODEL_H
#define MODEL_H
#include "layer.hpp"
#include <vector>

struct Model {
    std::vector<Layer> layers;

  public:
    Model() = default;

    ~Model() { clear(); }

  public:
    void init();
    void add_layer(size_t nb_inputs, size_t nb_nodes);
    void clear();
};

#endif
