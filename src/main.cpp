#include "matrix.hpp"
#include "mnist/minist_loader.hpp"
#include "model.hpp"
#include "trainer.hpp"
#include <cmath>
#include <iostream>
#include <random>

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
        auto [as, zs] = t.feedforward(elt.input);
        std::cout << "found: " << as.back()[0]
                  << "; expected: " << elt.ground_truth[0] << std::endl;
    }

    t.train(db, 4, nb_epochs, l_rate);

    std::cout << "after train:" << std::endl;
    for (auto const &elt : db) {
        auto [as, zs] = t.feedforward(elt.input);
        std::cout << "found: " << as.back()[0]
                  << "; expected: " << elt.ground_truth[0] << std::endl;
    }

    std::cout << "evaluation: " << t.evaluate(db) << std::endl;
}

void mnist_print_activation(Vector const &activation, Vector const &gt) {
    size_t max_idx = 0;
    size_t expected_solution = 0;

    std::cout << "activation = [ ";
    for (size_t i = 0; i < activation.size; ++i) {
        if (activation[i] > activation[max_idx]) {
            max_idx = i;
        }
        std::cout << activation[i] << " ";
    }
    for (size_t i = 0; i < gt.size; ++i) {
        if (gt[i] == 1) {
            expected_solution = i;
        }
    }
    std::cout << "] expected = " << expected_solution << " found = " << max_idx
              << std::endl;
}

void mnist_train_and_eval(Trainer t, DataBase const &train_db,
                          DataBase const &test_db, size_t nb_epochs,
                          double l_rate, size_t minibatch_size) {
    std::mt19937 gen(0);
    std::uniform_int_distribution<size_t> dist(0, test_db.size());

    t.train(train_db, minibatch_size, nb_epochs, l_rate);

    for (size_t i = 0; i < 10; ++i) {
        auto [as, zs] = t.feedforward(test_db[i].input);
        mnist_print_activation(as.back(), test_db[i].ground_truth);
    }
    std::cout << "average cost after training: " << t.evaluate(test_db)
              << std::endl;
}

int main(void) {
    MNISTLoader loader;
    Model m;
    Trainer t(&m, &sigmoid, &sigmoid_prime, &quadratic_loss,
              &quadratic_loss_prime);

    DataBase mnist_train_db =
        loader.load_db("../data/mnist/train-labels-idx1-ubyte",
                       "../data/mnist/train-images-idx3-ubyte");
    DataBase mnist_test_db =
        loader.load_db("../data/mnist/t10k-labels-idx1-ubyte",
                       "../data/mnist/t10k-images-idx3-ubyte");

    m.add_layer(28 * 28, 32);
    m.add_layer(32, 32);
    m.add_layer(32, 10);
    m.init(0);

    mnist_train_and_eval(t, mnist_train_db, mnist_test_db, 10'000, 0.006, 8);

    /* for (size_t i = 0; i < 1000; ++i) { */
    /*     t.train(mnist_train_db, 8, 1, 0.006); */
    /*     std::cout << "average cost after training: " */
    /*               << t.evaluate(mnist_test_db) << std::endl; */
    /* } */

    /* train_eval(t, OR_train, 100'000, 0.004); */
    /* train_eval(t, AND_train, 100'000, 0.004); */
    /* train_eval(t, XOR_train, 100'000, 0.004); */
    return 0;
}
