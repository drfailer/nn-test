#include "trainer.hpp"
#include "cblas.h"

using Vector = Trainer::Vector;
using Vectors = Trainer::Vectors;

Vector Trainer::compute_z(Layer const &layer, Vector const &a) {
    Vector z(layer.biases, layer.biases + layer.nb_nodes);

    // z = weights*a + biases
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.nb_nodes, layer.nb_inputs,
                1.0, layer.weights, layer.nb_inputs, a.data(), 1, 1.0, z.data(),
                1);
    return z;
}

Vector Trainer::apply(std::function<double(double)> fun, Vector const &vector) {
    Vector result(vector.size());

    for (size_t i = 0; i < vector.size(); ++i) {
        result[i] = fun(vector[i]);
    }
    return result;
}

Vector Trainer::apply(std::function<double(double, double)> fun,
                      Vector const &vector1, Vector const &vector2) {
    Vector result(vector1.size());

    assert(vector1.size() == vector2.size());
    for (size_t i = 0; i < vector1.size(); ++i) {
        result[i] = fun(vector1[i], vector2[i]);
    }
    return result;
}

Vector Trainer::act(Vector const &z) { return apply(act_, z); }

Vector Trainer::act_prime(Vector const &z) { return apply(act_prime_, z); }

Vector Trainer::cost(Vector const &ground_truth, Vector const &y) {
    return apply(cost_, ground_truth, y);
}

Vector Trainer::cost_prime(Vector const &ground_truth, Vector const &y) {
    return apply(cost_prime_, ground_truth, y);
}

Vector Trainer::hadamard(Vector const &a, Vector const &b) {
    Vector result(a.size());

    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

Vector Trainer::matmul(Vector err, Vector a) {
    Vector result(err.size() * a.size());

    for (size_t j = 0; j < err.size(); ++j) {
        for (size_t k = 0; k < a.size(); ++k) {
            result[j * err.size() + k] = err[j] * a[k];
        }
    }
    return result;
}

Vector Trainer::matmul(Layer const &layer, Vector &err) {
    Vector result(layer.nb_nodes);

    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer.nb_nodes, layer.nb_inputs,
                1.0, layer.weights, layer.nb_inputs, err.data(), 1, 0,
                result.data(), 1);
    return result;
}

std::pair<Vectors, Vectors> Trainer::feedforward(Vector const &input) {
    Vector z;
    Vectors zs = {};
    Vector a = input;
    Vectors as = {a};

    for (auto &layer : model_->layers) {
        z = compute_z(layer, a);
        zs.push_back(z);
        a = apply(act_, z);
        as.push_back(a);
    }
    return {as, zs};
}

std::pair<Vectors, Vectors> Trainer::backpropagate(Vector const &ground_truth,
                                                   Vectors as, Vectors zs) {
    Vector err =
        hadamard(cost_prime(ground_truth, as.back()), act_prime(zs.back()));
    Vectors grads_b(as.size());
    Vectors grads_w(as.size());

    grads_b[as.size() - 1] = err;
    grads_w[as.size() - 1] =
        hadamard(matmul(err, as[as.size() - 2]), act_prime(zs.back()));

    for (size_t l = model_->layers.size(); l >= 1; --l) {
        err = hadamard(matmul(model_->layers[l + 1], err), act_prime(zs[l]));
        grads_b[l] = err;
        grads_w[l] = matmul(err, act_prime(as[l - 1]));
    }
    return {grads_w, grads_b};
}
