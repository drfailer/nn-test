#include "model.hpp"
#include "trainer.hpp"
#include <cmath>
#include <iostream>

double OR_train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 1, 1},
    {1, 1, 1},
};

double sigmoid(double x) {
    return 1.0/(1.0+std::exp(-x));
}

double sigmoid_prime(double x) {
    return sigmoid(x)*(1-sigmoid(x));
}

double quadratic_loss(double y, double gt) {
    return 0.5*(gt-gt - y*y);
}

double quadratic_loss_prime(double y, double gt) {
    return gt - y;
}

int main(void) {
    Model m;
    Trainer t(&m, sigmoid, sigmoid_prime, quadratic_loss, quadratic_loss_prime);

    m.add_layer(2, 1);
    m.init();


    auto [as, zs] = t.feedforward({0, 1});
    std::cout << as.back()[0] << std::endl;

    m.clear();
    return 0;
}
