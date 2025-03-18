#ifndef TRAINER_H
#define TRAINER_H
#include "functions.hpp"
#include "minibatch_generator.hpp"
#include "model.hpp"
#include "types.hpp"
#include <cassert>
#include <cblas.h>

struct Tracer;

class Trainer {
  public:
    Trainer(Model *model, auto cost, auto activation, Tracer *tracer = nullptr)
        : model_(model), cost_(cost), activation_(activation), tracer_(tracer) {
    }

  public:
    Vector compute_z(Layer const &layer, Vector const &a) const;

    Vector act(Vector const &z) const;
    Vector act_prime(Vector const &z) const;
    Vector cost(Vector const &ground_truth, Vector const &y) const;
    Vector cost_prime(Vector const &ground_truth, Vector const &y) const;

    std::pair<Vectors, Vectors> feedforward(Vector const &input) const;
    std::pair<GradW, GradB> backpropagate(Vector const &ground_truth,
                                          Vectors const &as,
                                          Vectors const &zs) const;

    void update_minibatch(MinibatchGenerator const &minibatch,
                          double learning_rate);
    void update(DataSet const &ds, double learning_rate);

    void optimize(GradW const &grads_w, GradB const &grads_b,
                  double learning_rate);

    void train(DataSet const &ds, size_t nb_epochs, double learning_rate);
    void train_minibatch(DataSet const &ds, size_t minibatch_size,
                         size_t nb_epochs, double learning_rate,
                         uint32_t seed = 0);

    double evaluate_cost(DataSet const &test_ds) const;
    double evaluate_accuracy(DataSet const &test_ds) const;
    std::pair<double, double> evaluate(DataSet const &test_ds) const;

  private:
    Model *model_ = nullptr;
    CostFunction *cost_ = nullptr;
    ActivationFunction *activation_ = nullptr;
    OptimizeFunction *optimize_ = nullptr;
    Tracer *tracer_ = nullptr;

  public:
    void tracer(Tracer *tracer) { tracer_ = tracer; }

  private:
    int get_expected_label(Vector const &v) const;
};

#endif
