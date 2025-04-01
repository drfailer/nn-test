#include "trainer.hpp"
#include "cblas.h"
#include "tracer.hpp"
#include "types.hpp"
#include <cstring>
#include <numeric>

Vector Trainer::act(Vector const &z) const { return map(activation_, z); }

Vector Trainer::act_prime(Vector const &z) const {
    return map_derivative(activation_, z);
}

Vector Trainer::cost(Vector const &ground_truth, Vector const &y) const {
    return map(cost_, ground_truth, y);
}

Vector Trainer::cost_prime(Vector const &ground_truth, Vector const &y) const {
    return map_derivative(cost_, ground_truth, y);
}

Vector Trainer::compute_z(Layer const &layer, Vector const &a) const {
    assert(a.size == layer.nb_inputs);
    assert(layer.weights.rows == layer.nb_nodes);
    assert(layer.weights.cols == layer.nb_inputs);
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
    optimize_->execute(model_, grads_w, grads_b, learning_rate);
}

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

void Trainer::update(DataSet const &ds, double learning_rate) {
    for (size_t i = 0; i < ds.size(); ++i) {
        auto const &[x, gt] = ds[i];
        auto [as, zs] = feedforward(x);
        auto [grads_w, grads_b] = backpropagate(gt, as, zs);
        optimize(grads_w, grads_b, learning_rate);
    }
}

void Trainer::train(DataSet const &ds, size_t nb_epochs, double learning_rate) {
    if (tracer_) {
        tracer_->init(nb_epochs, ds.size(), learning_rate);
    }
    for (size_t epoch = 0; epoch < nb_epochs; ++epoch) {
        update(ds, learning_rate);
        if (tracer_) {
            tracer_->trace(this, epoch);
        }
    }
    if (tracer_) {
        tracer_->dump();
    }
}

void Trainer::train_minibatch(DataSet const &ds, size_t minibatch_size,
                              size_t nb_epochs, double learning_rate,
                              uint32_t seed) {
    assert(ds.size() >= minibatch_size);
    MinibatchGenerator minibatch(ds, minibatch_size, seed);

    if (tracer_) {
        tracer_->init(nb_epochs, minibatch_size, learning_rate);
    }
    for (size_t epoch = 0; epoch < nb_epochs; ++epoch) {
        minibatch.generate();
        update_minibatch(minibatch, learning_rate);
        if (tracer_) {
            tracer_->trace(this, epoch);
        }
    }
    if (tracer_) {
        tracer_->dump();
    }
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
