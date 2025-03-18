#include "trainer.hpp"
#include "cblas.h"
#include "types.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#define AVERAGE_MINIBATCH

/******************************************************************************/
/*                           trainer implementation */
/******************************************************************************/

Vector Trainer::act(Vector const &z) const { return map(act_, z); }

Vector Trainer::act_prime(Vector const &z) const { return map(act_prime_, z); }

Vector Trainer::cost(Vector const &ground_truth, Vector const &y) const {
    return map(cost_, ground_truth, y);
}

Vector Trainer::cost_prime(Vector const &ground_truth, Vector const &y) const {
    return map(cost_prime_, ground_truth, y);
}

Vector Trainer::compute_z(Layer const &layer, Vector const &a) const {
    assert(a.size == layer.nb_inputs);
    Vector z = layer.biases.clone();

    // z = weights*a + biases
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.nb_nodes, layer.nb_inputs,
                1.0, layer.weights.mem, layer.nb_inputs, a.mem, 1, 1.0, z.mem,
                1);
    return z;
}

std::pair<Vectors, Vectors> Trainer::feedforward(Vector const &input) const {
    Vectors zs = {};
    Vectors as = {input.clone()};

    for (auto const &layer : model_->layers) {
        zs.push_back(compute_z(layer, as.back()));
        as.push_back(act(zs.back()));
    }
    return {as, zs};
}

std::pair<GradW, GradB> Trainer::backpropagate(Vector const &ground_truth,
                                               Vectors const &as,
                                               Vectors const &zs) const {
    size_t L = model_->layers.size();
    auto &layers = model_->layers;
    Vector err =
        hadamard(cost_prime(ground_truth, as.back()), act_prime(zs.back()));
    GradB grads_b(L);
    GradW grads_w(L);

    grads_b[L - 1] = err.clone();
    grads_w[L - 1] = matmul(err, T(as[as.size() - 2]));

    for (size_t l = 2; l <= L; ++l) {
        err = hadamard(matmul(T(layers[L - l + 1].weights), err),
                       act_prime(zs[zs.size() - l]));
        grads_b[L - l] = err.clone();
        grads_w[L - l] = matmul(err, T(as[as.size() - l - 1]));
    }
    return {grads_w, grads_b};
}

// SGD -> we should have more in the future
void Trainer::optimize(GradW const &grads_w, GradB const &grads_b,
                       double const learning_rate) {
    for (size_t l = 0; l < model_->layers.size(); ++l) {
        assert(grads_w[l].rows == model_->layers[l].nb_nodes &&
               grads_w[l].cols == model_->layers[l].nb_inputs);
        assert(grads_b[l].size == model_->layers[l].nb_nodes);

        model_->layers[l].weights -= learning_rate * grads_w[l];
        model_->layers[l].biases -= learning_rate * grads_b[l];
    }
}

#ifdef AVERAGE_MINIBATCH
void Trainer::update_minibatch(MinibatchGenerator const &minibatch,
                               double learning_rate) {
    auto const &[x, gt] = minibatch.get(0);
    auto [as, zs] = feedforward(x);
    auto [total_grad_w, total_grad_b] = backpropagate(gt, as, zs);

    for (size_t i = 1; i < minibatch.size(); ++i) {
        auto const &[x, gt] = minibatch.get(i);
        auto [as, zs] = feedforward(x);
        auto [grads_w, grads_b] = backpropagate(gt, as, zs);
        total_grad_w += grads_w;
        total_grad_b += grads_b;
    }
    optimize(total_grad_w, total_grad_b,
             learning_rate / (double)minibatch.size());
}
#else
void Trainer::update_minibatch(MinibatchGenerator const &minibatch,
                               double learning_rate) {
    for (size_t i = 0; i < minibatch.size(); ++i) {
        auto const &[x, gt] = minibatch.get(i);
        auto [as, zs] = feedforward(x);
        auto [grads_w, grads_b] = backpropagate(gt, as, zs);
        optimize(grads_w, grads_b, learning_rate);
    }
}
#endif

void Trainer::train(DataSet const &ds, size_t minibatch_size, size_t nb_epochs,
                    double learning_rate, uint32_t seed) {
    assert(ds.size() >= minibatch_size);
    MinibatchGenerator minibatch(ds, minibatch_size, seed);

    for (size_t epoch = 0; epoch < nb_epochs; ++epoch) {
        minibatch.generate();
        update_minibatch(minibatch, learning_rate);
    }
}

/*
 * Train the model and dump the accuracy and the cost for the train and test
 * datasets at each epoch in the given file. The format is the following:
 * - nb_epochs minibatch_size learning_rate
 * - train costs
 * - train accuracy
 * - test costs
 * - test accuracy
 */
void Trainer::train_dump(std::string const &filename, DataSet const &train_ds,
                         DataSet const &test_ds, size_t minibatch_size,
                         size_t nb_epochs, double learning_rate,
                         uint32_t seed) {
    assert(train_ds.size() >= minibatch_size);
    MinibatchGenerator minibatch(train_ds, minibatch_size, seed);
    std::ofstream fs(filename, std::ios::binary);
    std::vector<double> costs_train(nb_epochs);
    std::vector<double> costs_test(nb_epochs);
    std::vector<double> accuracy_train(nb_epochs);
    std::vector<double> accuracy_test(nb_epochs);
    size_t loading_count = std::max<size_t>(1, nb_epochs / 100);

    fs.write(reinterpret_cast<char const *>(&nb_epochs), sizeof(size_t));
    fs.write(reinterpret_cast<char const *>(&minibatch_size), sizeof(size_t));
    fs.write(reinterpret_cast<char const *>(&learning_rate), sizeof(double));
    for (size_t epoch = 0; epoch < nb_epochs; ++epoch) {
        minibatch.generate();
        update_minibatch(minibatch, learning_rate);
        auto eval_train = evaluate(train_ds);
        auto eval_test = evaluate(test_ds);
        costs_train[epoch] = eval_train.first;
        costs_test[epoch] = eval_test.first;
        accuracy_train[epoch] = eval_train.second;
        accuracy_test[epoch] = eval_test.second;
        if (epoch % loading_count == 0) {
            std::cout << 100 * epoch / nb_epochs << " %" << std::endl;
        }
    }

    fs.write(reinterpret_cast<char const *>(costs_train.data()),
             nb_epochs * sizeof(double));
    fs.write(reinterpret_cast<char const *>(accuracy_train.data()),
             nb_epochs * sizeof(double));
    fs.write(reinterpret_cast<char const *>(costs_test.data()),
             nb_epochs * sizeof(double));
    fs.write(reinterpret_cast<char const *>(accuracy_test.data()),
             nb_epochs * sizeof(double));
}

int Trainer::get_expected_label(Vector const &v) const {
    int max_idx = 0;

    for (size_t i = 0; i < v.size; ++i) {
        if (v[i] > v[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

double Trainer::evaluate_cost(DataSet const &ds) const {
    double cost_sum = 0;

    for (auto const &elt : ds) {
        auto [as, zs] = feedforward(elt.input);
        auto costs = cost(elt.ground_truth, as.back());
        cost_sum += std::accumulate(costs.mem, costs.mem + costs.size, 0.0) /
                    costs.size;
    }
    return cost_sum / ds.size();
}

double Trainer::evaluate_accuracy(DataSet const &ds) const {
    size_t count_valid = 0;

    for (auto const &elt : ds) {
        auto [as, zs] = feedforward(elt.input);
        int found = get_expected_label(as.back());
        int expected = get_expected_label(elt.ground_truth);
        if (found == expected) {
            ++count_valid;
        }
    }
    return 100 * ((double)count_valid / (double)ds.size());
}

std::pair<double, double> Trainer::evaluate(DataSet const &ds) const {
    size_t count_valid = 0;
    double cost_sum = 0;
    double avg_cost = 0;
    double accuracy = 0;

    for (auto const &elt : ds) {
        auto [as, zs] = feedforward(elt.input);
        auto costs = cost(elt.ground_truth, as.back());
        int found = get_expected_label(as.back());
        int expected = get_expected_label(elt.ground_truth);

        if (found == expected) {
            ++count_valid;
        }
        cost_sum += std::accumulate(costs.mem, costs.mem + costs.size, 0.0) /
                    costs.size;
    }
    avg_cost = cost_sum / (double)ds.size();
    accuracy = 100 * ((double)count_valid / (double)ds.size());
    return {avg_cost, accuracy};
}
