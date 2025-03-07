#ifndef MODEL_H
#define MODEL_H
#include "layer.hpp"
#include <cstdint>
#include <vector>

struct Model {
    std::vector<Layer> layers;

  public:
    Model() = default;

    ~Model() { clear(); }

  public:
    void init(uint64_t seed);
    void input(size_t nb_inputs);
    void add_layer(size_t nb_nodes);
    void clear();

  private:
    size_t inputs_ = 0;
};

#endif
