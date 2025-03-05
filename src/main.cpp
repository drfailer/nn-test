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
DataBase XOR_train = {
    {{0, 0}, {0}},
    {{0, 1}, {1}},
    {{1, 0}, {1}},
    {{1, 1}, {0}},
};

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double binary_step(double x) {
    if (x < 0) {
        return 0;
    }
    return 1;
}

double binary_step_prime(double) { return 0; }

double sigmoid_prime(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

double quadratic_loss(double gt, double y) {
    double diff = y - gt;
    return 0.5 * diff * diff;
}

double quadratic_loss_prime(double gt, double y) { return y - gt; }

void train_eval(Trainer t, DataBase const &db, size_t nb_epochs,
                double l_rate) {
    std::cout << "start value:" << std::endl;
    for (auto const &elt : db) {
        auto [as, zs] = t.feedforward(elt.first);
        std::cout << "found: " << as.back()[0]
                  << "; expected: " << elt.second[0] << std::endl;
    }

    t.train(db, 4, nb_epochs, l_rate);

    std::cout << "after train:" << std::endl;
    for (auto const &elt : db) {
        auto [as, zs] = t.feedforward(elt.first);
        std::cout << "found: " << as.back()[0]
                  << "; expected: " << elt.second[0] << std::endl;
    }

    std::cout << "evaluation: " << t.evaluate(db) << std::endl;
}

int main(void) {
    Model m;
    Trainer t(&m, &sigmoid, &sigmoid_prime, &quadratic_loss,
              &quadratic_loss_prime);

    m.add_layer(2, 2);
    m.add_layer(2, 1);
    m.init(0);

    train_eval(t, OR_train, 100'000, 0.004);
    train_eval(t, AND_train, 100'000, 0.004);
    train_eval(t, XOR_train, 100'000, 0.004);

    m.clear();
    return 0;
}
