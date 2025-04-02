#include "math.hpp"
#include "mnist/minist_loader.hpp"
#include "model.hpp"
#include "tracer.hpp"
#include "trainer.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

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

template <typename Cost, typename Act, typename Opt>
void train_eval(DataSet const &ds, size_t nb_epochs, ftype l_rate) {
    Model m;
    Cost quadratic_loss;
    Act sigmoid;
    Opt sgd;
    Trainer t(&m, &quadratic_loss, &sigmoid, &sgd);

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

void train_logic_gate() {
    train_eval<QuadraticLoss, Sigmoid, SGD>(OR_train, 100'000, 0.004);
    train_eval<QuadraticLoss, Sigmoid, SGD>(AND_train, 100'000, 0.004);
    train_eval<QuadraticLoss, Sigmoid, SGD>(XOR_train, 100'000, 0.004);
}

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

Model create_mnist_model() {
    Model m;
    m.input(28 * 28);
    m.add_layer(32);
    m.add_layer(10);
    m.init(0);
    return m;
}

template <typename Cost, typename Act, typename Opt>
void test_mnist(DataSet const &train_data, DataSet const &test_data,
                size_t nb_epochs, ftype learning_rate,
                size_t minibatch_size = 0) {
    Model m = create_mnist_model();
    Cost cost;
    Act act;
    Opt opt;
    Trainer t(&m, &cost, &act, &opt);
    Tracer tracer(train_data, test_data);

    mnist_train_and_eval(t, train_data, test_data, nb_epochs, learning_rate,
                         minibatch_size);
}

template <typename Cost, typename Act, typename Opt>
void trace_mnist(DataSet const &train_data, DataSet const &test_data,
                 size_t nb_epochs, ftype learning_rate,
                 size_t minibatch_size = 0) {
    Model m = create_mnist_model();
    Cost cost;
    Act act;
    Opt opt;
    Trainer t(&m, &cost, &act, &opt);
    Tracer tracer(train_data, test_data);

    t.tracer(&tracer);
    if (minibatch_size == 0) {
        t.train(train_data, nb_epochs, learning_rate);
    } else {
        t.train_minibatch(train_data, minibatch_size, nb_epochs, learning_rate);
    }
}

int main(void) {
    MNISTLoader loader;
    DataSet mnist_train_data =
        loader.load_ds("../data/mnist/train-labels-idx1-ubyte",
                       "../data/mnist/train-images-idx3-ubyte");
    DataSet mnist_test_data =
        loader.load_ds("../data/mnist/t10k-labels-idx1-ubyte",
                       "../data/mnist/t10k-images-idx3-ubyte");
    test_compute_z();
    test_vector();
    trace_mnist<QuadraticLoss, Sigmoid, SGD>(mnist_train_data, mnist_test_data,
                                             1'000, 0.01, 8);
    trace_mnist<QuadraticLoss, Sigmoid, SGD>(mnist_train_data, mnist_test_data,
                                             30, 0.01);
    test_mnist<QuadraticLoss, Sigmoid, SGD>(mnist_train_data, mnist_test_data,
                                            1, 0.01);
    test_mnist<QuadraticLoss, Sigmoid, SGD>(mnist_train_data, mnist_test_data,
                                            30, 0.01);
    test_mnist<QuadraticLoss, Sigmoid, SGD>(mnist_train_data, mnist_test_data,
                                            100'000, 8);
    return 0;
}
