#include "math.hpp"
#include "mnist/minist_loader.hpp"
#include "model.hpp"
#include "trainer.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include "tracer.hpp"

DataSet OR_train = {
    {{0, 0}, {0}},
    {{0, 1}, {1}},
    {{1, 0}, {1}},
    {{1, 1}, {1}},
};
DataSet AND_train = {
    {{0, 0}, {0}},
    {{0, 1}, {0}},
    {{1, 0}, {0}},
    {{1, 1}, {1}},
};
DataSet XOR_train = {
    {{0, 0}, {0}},
    {{0, 1}, {1}},
    {{1, 0}, {1}},
    {{1, 1}, {0}},
};

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
    Sigmoid sigmoid;
    QuadraticLoss quadratic_loss;
    SGD sgd;
    Trainer t(&m, &quadratic_loss, &sigmoid, &sgd);
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

    Vector z = t.compute_z(Layer{w, b, 2, 2}, a);
    assert(211 == z[0]);
    assert(432 == z[1]);
}

void train_eval(Trainer t, DataSet const &ds, size_t nb_epochs,
                ftype l_rate) {
    std::cout << "start value:" << std::endl;
    for (auto const &elt : ds) {
        auto [as, zs] = t.feedforward(elt.input);
        std::cout << "found: " << as.back()[0]
                  << "; expected: " << elt.ground_truth[0] << std::endl;
    }

    t.train_minibatch(ds, 4, nb_epochs, l_rate);

    std::cout << "after train:" << std::endl;
    for (auto const &elt : ds) {
        auto [as, zs] = t.feedforward(elt.input);
        std::cout << "found: " << as.back()[0]
                  << "; expected: " << elt.ground_truth[0] << std::endl;
    }

    std::cout << "evaluation: " << t.evaluate_cost(ds) << std::endl;
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

void mnist_train_and_eval(Trainer t, DataSet const &train_ds,
                          DataSet const &test_ds, size_t nb_epochs,
                          ftype l_rate, size_t minibatch_size) {
    std::mt19937 gen(0);

    auto t1 = std::chrono::system_clock::now();
    if (minibatch_size == 0) {
        t.train(train_ds, nb_epochs, l_rate);
    } else {
        t.train_minibatch(train_ds, minibatch_size, nb_epochs, l_rate);
    }
    auto t2 = std::chrono::system_clock::now();
    std::cout << "training time : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << std::endl;

    for (size_t i = 0; i < 10; ++i) {
        auto [as, zs] = t.feedforward(test_ds[i].input);
        MNISTLoader::print_image(test_ds[i].input, 28, 28);
        mnist_print_activation(as.back(), test_ds[i].ground_truth);
    }
    auto eval = t.evaluate(test_ds);
    std::cout << "average cost after training: " << eval.first << std::endl;
    std::cout << "evaluation: " << eval.second << "%" << std::endl;
}

int main(void) {
    MNISTLoader loader;
    Model m;
    Sigmoid sigmoid;
    QuadraticLoss quadratic_loss;
    SGD sgd;
    Trainer t(&m, &quadratic_loss, &sigmoid, &sgd);
    /* Adam adam; */
    /* Trainer t(&m, &quadratic_loss, &sigmoid, &adam); */
    std::random_device r;

    test_compute_z();
    test_vector();
    /* return 0; */

    DataSet mnist_train_ds =
        loader.load_ds("../data/mnist/train-labels-idx1-ubyte",
                       "../data/mnist/train-images-idx3-ubyte");
    DataSet mnist_test_ds =
        loader.load_ds("../data/mnist/t10k-labels-idx1-ubyte",
                       "../data/mnist/t10k-images-idx3-ubyte");
    Tracer tracer(mnist_train_ds, mnist_test_ds);

    m.input(28 * 28);
    /* m.add_layer(32); */
    m.add_layer(10);

    m.init(r());

    // this learns fast without the AVERAGE_MINIBATCH
    mnist_train_and_eval(t, mnist_train_ds, mnist_test_ds, 1, 0.01, 0);
    /* mnist_train_and_eval(t, mnist_train_ds, mnist_test_ds, 30, 0.01, 0); */
    /* mnist_train_and_eval(t, mnist_train_ds, mnist_test_ds, 30 * 60'000, 0.01, 1); */
    /* mnist_train_and_eval(t, mnist_train_ds, mnist_test_ds, 50'000, 1, 8); */

    /* t.tracer(&tracer); */
    /* t.train_minibatch(mnist_train_ds, 8, 1'000, 1); */
    /* t.train_minibatch(mnist_train_ds, 8, 1'000, 0.1); */
    /* t.train(mnist_train_ds, 30, 0.01); */

    /* t.train(mnist_train_ds, 8, 100, 0.002, r()); */
    /* std::cout << "evaluation (train): " << mnist_eval(t, mnist_train_ds) */
    /*           << std::endl; */
    /* std::cout << "evaluation (test): " << mnist_eval(t, mnist_test_ds) */
    /*           << std::endl; */

    /* for (size_t i = 0; i < 1000; ++i) { */
    /*     t.train(mnist_train_ds, 8, 100, 0.002, r()); */
    /*     std::cout << "evaluation (train): " << mnist_eval(t, mnist_train_ds) */
    /*               << ", loss = " << t.evaluate(mnist_train_ds) << std::endl; */
    /*     std::cout << "evaluation (test): " << mnist_eval(t, mnist_test_ds) */
    /*               << ", loss = " << t.evaluate(mnist_test_ds) << std::endl; */
    /* } */

    /* train_eval(t, OR_train, 100'000, 0.004); */
    /* train_eval(t, AND_train, 100'000, 0.004); */
    /* train_eval(t, XOR_train, 100'000, 0.004); */
    return 0;
}
