#include "model.hpp"

double OR_train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 1, 1},
    {1, 1, 1},
};

int main(void) {
    Model m;

    m.add_layer(2, 1);
    m.init();
    m.clear();
    return 0;
}
