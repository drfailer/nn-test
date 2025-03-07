#include "trainer.hpp"
#include "cblas.h"
#include "types.hpp"
#include <cstring>
#include <numeric>
#define AVERAGE_MINIBATCH

/******************************************************************************/
/*                                  helpers                                   */
/******************************************************************************/

GradW const &operator+=(GradW &lhs, GradW const &rhs) {
    assert(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

GradB const &operator+=(GradB &lhs, GradB const &rhs) {
    assert(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

GradW const &operator/=(GradW &lhs, double constant) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] /= constant;
    }
    return lhs;
}

GradB const &operator/=(GradB &lhs, double constant) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] /= constant;
    }
    return lhs;
}

/******************************************************************************/
/*                           trainer implementation */
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
    total_grad_w /= (double)minibatch.size();
    total_grad_b /= (double)minibatch.size();

    optimize(total_grad_w, total_grad_b, learning_rate);
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

void Trainer::train(DataBase const &db, size_t minibatch_size, size_t nb_epochs,
                    double learning_rate, uint32_t seed) {
    assert(db.size() >= minibatch_size);
    MinibatchGenerator minibatch(minibatch_size, seed);

    for (size_t epoch = 0; epoch < nb_epochs; ++epoch) {
        minibatch.generate(db);
        update_minibatch(minibatch, learning_rate);
    }
}

double Trainer::evaluate(DataBase const &test_db) {
    double result = 0;

    for (auto const &elt : test_db) {
        auto [as, zs] = feedforward(elt.input);
        auto costs = cost(elt.ground_truth, as.back());
        result += std::accumulate(costs.mem, costs.mem + costs.size, 0.0) /
                  costs.size;
    }
    return result / test_db.size();
}
