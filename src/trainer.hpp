#ifndef TRAINER_H
#define TRAINER_H
#include "model.hpp"
#include "types.hpp"
#include <cassert>
#include <cblas.h>
#include <functional>

class Trainer {
  public:
    Trainer(Model *model, auto act, auto act_prime, auto cost, auto cost_prime)
        : model_(model), act_(act), act_prime_(act_prime), cost_(cost),
          cost_prime_(cost_prime) {}

  public:
    Vector compute_z(Layer const &layer, Vector const &a);

    Vector act(Vector const &z);
    Vector act_prime(Vector const &z);
    Vector cost(Vector const &ground_truth, Vector const &y);
    Vector cost_prime(Vector const &ground_truth, Vector const &y);

    std::pair<Vectors, Vectors> feedforward(Vector const &input);
    std::pair<Vectors, Vectors> backpropagate(Vector const &ground_truth,
                                              Vectors const &as,
                                              Vectors const &zs);

    void update_minibatch(DataBase const &minibatch, double learning_rate);
    void optimize(Vectors const &grads_w, Vectors const &grads_b,
                  double learning_rate);
    void train(DataBase const &db, size_t minibatch_size, size_t nb_epochs,
               double learning_rate);
    double evaluate(DataBase const &test_db);

  private:
    Model *model_ = nullptr;
    std::function<double(double)> act_;
    std::function<double(double)> act_prime_;
    std::function<double(double, double)> cost_;
    std::function<double(double, double)> cost_prime_;
};

#endif
