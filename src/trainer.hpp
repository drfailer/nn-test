#ifndef TRAINER_H
#define TRAINER_H
#include "model.hpp"
#include <cassert>
#include <cblas.h>
#include <functional>

class Trainer {
  public:
    Trainer(Model *model, auto act, auto act_prime, auto cost, auto cost_prime)
        : model_(model), act_(act), act_prime_(act_prime), cost_(cost),
          cost_prime_(cost_prime) {}

  public:
    using Vector = std::vector<double>;
    using Vectors = std::vector<std::vector<double>>;

  public:
    Vector compute_z(Layer const &layer, Vector const &a);

    Vector apply(std::function<double(double)> fun, Vector const &vector);
    Vector apply(std::function<double(double, double)> fun, Vector const &vector1, Vector const &vector2);

    Vector act(Vector const &z);
    Vector act_prime(Vector const &z);
    Vector cost(Vector const &ground_truth, Vector const &y);
    Vector cost_prime(Vector const &ground_truth, Vector const &y);

    Vector hadamard(Vector const &a, Vector const &b);
    Vector matmul(Vector err, Vector a);
    Vector matmul(Layer const &layer, Vector &err);

    std::pair<Vectors, Vectors> feedforward(Vector const &input);
    std::pair<Vectors, Vectors> backpropagate(Vector const &ground_truth, Vectors as, Vectors zs);

  private:
    Model *model_;
    std::function<double(double)> act_;
    std::function<double(double)> act_prime_;
    std::function<double(double, double)> cost_;
    std::function<double(double, double)> cost_prime_;
};

#endif
