#include "trainer.hpp"
#include "cblas.h"
#include <cstring>
#include <iostream>
#include <numeric>

/******************************************************************************/
/*                           trainer implementation                           */
/******************************************************************************/

Vector Trainer::act(Vector const &z) { return map(act_, z); }

Vector Trainer::act_prime(Vector const &z) { return map(act_prime_, z); }

Vector Trainer::cost(Vector const &ground_truth, Vector const &y) {
    return map(cost_, ground_truth, y);
}

Vector Trainer::cost_prime(Vector const &ground_truth, Vector const &y) {
    return map(cost_prime_, ground_truth, y);
}

Vector Trainer::compute_z(Layer const &layer, Vector const &a) {
    assert(a.size == layer.nb_inputs);
    Vector z = layer.biases.clone();

    // z = weights*a + biases
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.nb_nodes, layer.nb_inputs,
                1.0, layer.weights.mem, layer.nb_inputs, a.mem, 1, 1.0, z.mem,
                1);
    return z;
}

std::pair<Vectors, Vectors> Trainer::feedforward(Vector const &input) {
    Vectors zs = {};
    Vectors as = {input.clone()};

    for (auto const &layer : model_->layers) {
        zs.emplace_back(compute_z(layer, as.back()));
        as.emplace_back(act(zs.back()));
    }
    return {as, zs};
}

std::pair<GradW, GradB> Trainer::backpropagate(Vector const &ground_truth,
                                               Vectors const &as,
                                               Vectors const &zs) {
    size_t L = model_->layers.size();
    auto &layers = model_->layers;
    Vector err =
        hadamard(cost_prime(ground_truth, as.back()), act_prime(zs.back()));
    GradB grads_b(L);
    GradW grads_w(L);

    grads_b[L - 1] = err.clone();
    grads_w[L - 1] = matmul(err, as[as.size() - 2]);

    for (size_t l = 2; l <= L; ++l) {
        err = hadamard(matmul(layers[L - l + 1].weights, err),
                       act_prime(zs[zs.size() - l]));
        grads_b[L - l] = err.clone();
        grads_w[L - l] = matmul(err, as[as.size() - l - 1]);
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

void Trainer::update_minibatch(DataBase const &minibatch,
                               double learning_rate) {
    // TODO: the grads_w and grads_b should be averaged
    for (auto const &[x, gt] : minibatch) {
        auto [as, zs] = feedforward(x);
        auto [grads_w, grads_b] = backpropagate(gt, as, zs);
        optimize(grads_w, grads_b, learning_rate);
    }
}

void Trainer::train(DataBase const &db, size_t minibatch_size, size_t nb_epochs,
                    double learning_rate) {
    for (size_t epoch = 0; epoch < nb_epochs; ++epoch) {
        // TODO: take a random sample of the db to build the minibatch
        update_minibatch(db, learning_rate);
    }
}

double Trainer::evaluate(DataBase const &test_db) {
    double result = 0;

    for (auto const &elt : test_db) {
        auto [as, zs] = feedforward(elt.first);
        auto costs = cost(elt.second, as.back());
        result += std::accumulate(costs.mem, costs.mem + costs.size, 0.0) /
                  costs.size;
    }
    return result / test_db.size();
}
