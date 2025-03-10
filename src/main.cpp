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

double sigmoid_prime(double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }

double binary_step(double x) {
    if (x < 0) {
        return 0;
    }
    return 1;
}

double binary_step_prime(double) { return 0; }

double quadratic_loss(double gt, double y) {
    double diff = gt - y;
    return 0.5 * diff * diff;
}

double quadratic_loss_prime(double gt, double y) { return y - gt; }

void test_vector() {
    Vector v1 = {1, 2};
    assert(1 == v1[0]);
    assert(2 == v1[1]);

    Vector v2 = v1;
    assert(v2[0] == v1[0]);
    assert(v2[1] == v1[1]);

    Vector v3 = v1.clone();
    assert(v3[0] == v1[0]);
    assert(v3[1] == v1[1]);
}

void test_compute_z() {
    Model m;
    Trainer t(&m, &sigmoid, &sigmoid_prime, &quadratic_loss,
              &quadratic_loss_prime);
    Matrix w(2, 2);
    Vector b(2);
    Vector a(2);

    w[0][0] = 1;
    w[0][1] = 2;
    w[1][0] = 3;
    w[1][1] = 4;

    b[0] = 1;
    b[1] = 2;

    a[0] = 10;
    a[1] = 100;

    Vector z = t.compute_z(Layer{ w, b, 2, 2 }, a);
    assert(211 == z[0]);
    assert(432 == z[1]);
}

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

int get_label(Vector const &v) {
    int max_idx = 0;

    for (size_t i = 0; i < v.size; ++i) {
        if (v[i] > v[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

void mnist_print_activation(Vector const &activation, Vector const &gt) {
    std::cout << "activation = [ ";
    for (size_t i = 0; i < activation.size; ++i) {
        std::cout << activation[i] << " ";
    }
    std::cout << "] expected = " << get_label(gt)
              << " found = " << get_label(activation) << std::endl;
}

double mnist_eval(Trainer t, DataBase const &test_db) {
    size_t count_valid = 0;

    for (auto const &elt : test_db) {
        auto [as, zs] = t.feedforward(elt.input);
        int found = get_label(as.back());
        int expected = get_label(elt.ground_truth);
        if (found == expected) {
            ++count_valid;
        }
    }
    return 100 * ((double)count_valid / (double)test_db.size());
}

void mnist_train_and_eval(Trainer t, DataBase const &train_db,
                          DataBase const &test_db, size_t nb_epochs,
                          double l_rate, size_t minibatch_size) {
    std::mt19937 gen(0);

    t.train(train_db, minibatch_size, nb_epochs, l_rate);

    for (size_t i = 0; i < 10; ++i) {
        auto [as, zs] = t.feedforward(test_db[i].input);
        MNISTLoader::print_image(test_db[i].input, 28, 28);
        mnist_print_activation(as.back(), test_db[i].ground_truth);
    }
    std::cout << "average cost after training: " << t.evaluate(test_db)
              << std::endl;
    std::cout << "evaluation: " << mnist_eval(t, test_db) << "%" << std::endl;
}

int main(void) {
    MNISTLoader loader;
    Model m;
    Trainer t(&m, &sigmoid, &sigmoid_prime, &quadratic_loss,
              &quadratic_loss_prime);
    std::random_device r;

    test_compute_z();
    test_vector();
    /* return 0; */

    DataBase mnist_train_db =
        loader.load_db("../data/mnist/train-labels-idx1-ubyte",
                       "../data/mnist/train-images-idx3-ubyte");
    DataBase mnist_test_db =
        loader.load_db("../data/mnist/t10k-labels-idx1-ubyte",
                       "../data/mnist/t10k-images-idx3-ubyte");

    m.input(28 * 28);
    m.add_layer(10);

    m.init(r());

    mnist_train_and_eval(t, mnist_train_db, mnist_test_db, 10'000, 0.002, 8);

    /* t.train(mnist_train_db, 8, 100, 0.002, r()); */
    /* std::cout << "evaluation (train): " << mnist_eval(t, mnist_train_db) */
    /*           << std::endl; */
    /* std::cout << "evaluation (test): " << mnist_eval(t, mnist_test_db) */
    /*           << std::endl; */

    /* for (size_t i = 0; i < 1000; ++i) { */
    /*     t.train(mnist_train_db, 8, 100, 0.002, r()); */
    /*     std::cout << "evaluation (train): " << mnist_eval(t, mnist_train_db) */
    /*               << std::endl; */
    /*     std::cout << "evaluation (test): " << mnist_eval(t, mnist_test_db) */
    /*               << std::endl; */
    /* } */

    /* train_eval(t, OR_train, 100'000, 0.004); */
    /* train_eval(t, AND_train, 100'000, 0.004); */
    /* train_eval(t, XOR_train, 100'000, 0.004); */
    return 0;
}
