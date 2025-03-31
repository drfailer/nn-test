#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include "math.hpp"
#include "model.hpp"
#include <cassert>
#include <cmath>

/******************************************************************************/
/*                                 interfaces                                 */
/******************************************************************************/

struct CostFunction {
    virtual double execute(double ground_truth, double layer_output) = 0;
    virtual double derivative(double ground_truth, double layer_output) = 0;
};

struct ActivationFunction {
    virtual double execute(double) = 0;
    virtual double derivative(double) = 0;
};

struct OptimizeFunction {
    virtual void execute(Model *model, GradW grads_w, GradB grads_b,
                         double learning_rate) = 0;
};

/******************************************************************************/
/*                              implementations                               */
/******************************************************************************/

struct QuadraticLoss : CostFunction {
    double execute(double ground_truth, double output) override {
        double diff = ground_truth - output;
        return 0.5 * diff * diff;
    }

    double derivative(double ground_truth, double output) override {
        return output - ground_truth;
    }
};

struct Sigmoid : ActivationFunction {
    double execute(double x) override { return 1.0 / (1.0 + std::exp(-x)); }

    double derivative(double x) override {
        return execute(x) * (1.0 - execute(x));
    }
};

struct SGD : OptimizeFunction {
    void execute(Model *model, GradW grads_w, GradB grads_b,
                 double learning_rate) override {
        for (size_t l = 0; l < model->layers.size(); ++l) {
            assert(grads_w[l].rows == model->layers[l].nb_nodes &&
                   grads_w[l].cols == model->layers[l].nb_inputs);
            assert(grads_b[l].size == model->layers[l].nb_nodes);

            model->layers[l].weights -= learning_rate * grads_w[l];
            model->layers[l].biases -= learning_rate * grads_b[l];
        }
    }
};

/******************************************************************************/
/*                                 functions                                  */
/******************************************************************************/

Vector map(ActivationFunction *act, Vector const &v);
Vector map_derivative(ActivationFunction *act, Vector const &v);
Vector map(CostFunction *cost, Vector const &v1, Vector const &v2);
Vector map_derivative(CostFunction *cost, Vector const &v1, Vector const &v2);

#endif
