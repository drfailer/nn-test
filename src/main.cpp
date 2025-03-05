#include "model.hpp"
#include "trainer.hpp"
#include <cmath>
#include <iostream>

DataBase OR_train = {
    {{0, 0}, {0}},
    {{0, 1}, {1}},
    {{1, 0}, {1}},
    {{1, 1}, {1}},
};
DataBase AND_train = {
    {{0, 0}, {0}},
    {{0, 1}, {0}},
    {{1, 0}, {0}},
    {{1, 1}, {1}},
};
DataBase train = OR_train;
/* DataBase train = AND_train; */


double sigmoid(double x) {
    return 1.0/(1.0+std::exp(-x));
}

double binary_step(double x) {
    if (x < 0) {
        return 0;
    }
    return 1;
}

double binary_step_prime(double) {
    return 0;
}

double sigmoid_prime(double x) {
    return sigmoid(x)*(1-sigmoid(x));
}

double quadratic_loss(double gt, double y) {
    double diff = y - gt;
    return 0.5*diff*diff;
}

double quadratic_loss_prime(double gt, double y) {
    return y - gt;
}

int main(void) {
    Model m;

    m.add_layer(2, 1);
    m.init();

    Trainer t(&m, &sigmoid, &sigmoid_prime, &quadratic_loss, &quadratic_loss_prime);
    /* Trainer t(&m, binary_step, binary_step, quadratic_loss, quadratic_loss_prime); */


    std::cout << "start value:" << std::endl;
    for (auto const &elt : train) {
        auto [as, zs] = t.feedforward(elt.first);
        std::cout << "found: " << as.back()[0] << "; expected: " << elt.second[0] << std::endl;
    }

    t.train(train, 4, 100'000, 0.004);

    std::cout << "after train:" << std::endl;
    for (auto const &elt : train) {
        auto [as, zs] = t.feedforward(elt.first);
        std::cout << "found: " << as.back()[0] << "; expected: " << elt.second[0] << std::endl;
    }

    m.clear();
    return 0;
}
