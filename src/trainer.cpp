#include "trainer.hpp"
#include "cblas.h"
#include <cstring>
#include <iostream>
#include <numeric>

/******************************************************************************/
/*                              helper function                               */
/******************************************************************************/

Vector hadamard(Vector const &a, Vector const &b) {
    Vector result(a.size());

    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

// multiply the vectors to create a square matrix
Vector matmul(Vector err, Vector a) {
    Vector result(err.size() * a.size());

    for (size_t j = 0; j < err.size(); ++j) {
        for (size_t k = 0; k < a.size(); ++k) {
            result[j * a.size() + k] = err[j] * a[k];
        }
    }
    return result;
}

Vector matmul(Layer const &layer, Vector &err) {
    Vector result(layer.nb_inputs, 0);

    // BUG: there is an issue with cblas (causing memory leak and bad access)
    cblas_dgemv(CblasRowMajor, CblasTrans, layer.nb_nodes, layer.nb_inputs, 1.0,
                layer.weights, layer.nb_inputs, err.data(), 1, 0, result.data(),
                1);
    return result;
}

Vector apply(std::function<double(double)> fun, Vector const &vector) {
    Vector result(vector.size());

    for (size_t i = 0; i < vector.size(); ++i) {
        result[i] = fun(vector[i]);
    }
    return result;
}

Vector apply(std::function<double(double, double)> fun,
                      Vector const &vector1, Vector const &vector2) {
    Vector result(vector1.size());

    assert(vector1.size() == vector2.size());
    for (size_t i = 0; i < vector1.size(); ++i) {
        result[i] = fun(vector1[i], vector2[i]);
    }
    return result;
}

struct LazyCVMult {
    double constant;
    Vector const &vector;
};

LazyCVMult operator*(double constant, Vector const &vector) {
    return LazyCVMult(constant, vector);
}

double const *operator-=(double *v, LazyCVMult const &cvmult) {
    for (size_t i = 0; i < cvmult.vector.size(); ++i) {
        v[i] -= cvmult.constant * cvmult.vector[i];
    }
    return v;
}

/******************************************************************************/
/*                           trainer implementation                           */
/******************************************************************************/

Vector Trainer::act(Vector const &z) { return apply(act_, z); }

Vector Trainer::act_prime(Vector const &z) { return apply(act_prime_, z); }

Vector Trainer::cost(Vector const &ground_truth, Vector const &y) {
    return apply(cost_, ground_truth, y);
}

Vector Trainer::cost_prime(Vector const &ground_truth, Vector const &y) {
    return apply(cost_prime_, ground_truth, y);
}

Vector Trainer::compute_z(Layer const &layer, Vector const &a) {
    assert(a.size() == layer.nb_inputs);
    Vector z(layer.biases, layer.biases + layer.nb_nodes);

    // z = weights*a + biases
    // BUG: there is an issue with cblas (causing memory leak and bad access)
    /* cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.nb_nodes, layer.nb_inputs, */
    /*             1.0, layer.weights, layer.nb_inputs, a.data(), 1, 1.0, z.data(), */
    /*             1); */
    for (size_t i = 0; i < layer.nb_nodes; ++i) {
        for (size_t j = 0; j < layer.nb_inputs; ++j) {
            z[i] += layer.weights[i * layer.nb_inputs + j] * a[j];
        }
    }
    return z;
}

std::pair<Vectors, Vectors> Trainer::feedforward(Vector const &input) {
    Vector z;
    Vectors zs = {};
    Vector a = input;
    Vectors as = {a};

    for (auto const &layer : model_->layers) {
        z = compute_z(layer, a);
        zs.push_back(z);
        a = act(z);
        as.push_back(a);
    }
    return {as, zs};
}

std::pair<Vectors, Vectors> Trainer::backpropagate(Vector const &ground_truth,
                                                   Vectors const &as,
                                                   Vectors const &zs) {
    size_t L = model_->layers.size();
    Vector err =
        hadamard(cost_prime(ground_truth, as.back()), act_prime(zs.back()));
    Vectors grads_b(L);
    Vectors grads_w(L);

    grads_b[L - 1] = err;
    grads_w[L - 1] = matmul(err, as[as.size() - 2]);

    for (size_t l = 2; l <= L; ++l) {
        err = hadamard(matmul(model_->layers[L - l + 1], err),
                       act_prime(zs[zs.size() - l]));
        grads_b[L - l] = err;
        grads_w[L - l] = matmul(err, as[as.size() - l - 1]);
    }
    return {grads_w, grads_b};
}

// SGD -> we should have more in the future
void Trainer::optimize(Vectors const &grads_w, Vectors const &grads_b,
                       double const learning_rate) {
    for (size_t l = 0; l < model_->layers.size(); ++l) {
        assert(grads_w[l].size() ==
               model_->layers[l].nb_nodes * model_->layers[l].nb_inputs);
        assert(grads_b[l].size() == model_->layers[l].nb_nodes);

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
        result +=
            std::accumulate(costs.begin(), costs.end(), 0.0) / costs.size();
    }
    return result / test_db.size();
}
