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
    virtual void execute(Model *model, GradW const &grads_w,
                         GradB const &grads_b, double learning_rate) = 0;
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
    void execute(Model *model, GradW const &grads_w, GradB const &grads_b,
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

struct Adam : OptimizeFunction {
    /* Create m and v and set the memory to 0 */
    void init(GradW const &grads_w, GradB const &grads_b) {
        m_w.resize(grads_w.size());
        v_w.resize(grads_w.size());
        m_b.resize(grads_b.size());
        v_b.resize(grads_b.size());

        for (size_t l = 0; l < grads_w.size(); ++l) {
            m_w[l] = Matrix(grads_w[l].rows, grads_w[l].cols);
            memset(m_w[l].mem, 0,
                   grads_w[l].rows * grads_w[l].cols * sizeof(*grads_w[l].mem));
            v_w[l] = Matrix(grads_w[l].rows, grads_w[l].cols);
            memset(v_w[l].mem, 0,
                   grads_w[l].rows * grads_w[l].cols * sizeof(*grads_w[l].mem));
        }

        for (size_t l = 0; l < grads_b.size(); ++l) {
            m_b[l] = Vector(grads_b[l].size);
            memset(m_b[l].mem, 0, grads_b[l].size * sizeof(*grads_b[l].mem));
            v_b[l] = Vector(grads_b[l].size);
            memset(v_b[l].mem, 0, grads_b[l].size * sizeof(*grads_b[l].mem));
        }
        b1_t = b1;
        b2_t = b2;
        is_init = true;
    }

    // m = b1 * m + (1 - b1) * grads
    // v = b2 * v + (1 - b2) * grads * grads
    void compute_mv(GradW const &grads_w, GradB const &grads_b) {
        for (size_t l = 0; l < grads_w.size(); ++l) {
            double *mw = m_w[l].mem;
            double *vw = v_w[l].mem;
            double *gw = grads_w[l].mem;

            for (size_t i = 0; i < grads_w[l].rows * grads_w[l].cols; ++i) {
                mw[i] = b1 * mw[i] + (1 - b1) * gw[i];
                vw[i] = b2 * vw[i] + (1 - b2) * gw[i] * gw[i];
            }
        }

        for (size_t l = 0; l < grads_b.size(); ++l) {
            double *mb = m_b[l].mem;
            double *vb = v_b[l].mem;
            double *gb = grads_b[l].mem;

            for (size_t i = 0; i < grads_b[l].size; ++i) {
                mb[i] = b1 * mb[i] + (1 - b1) * gb[i];
                vb[i] = b2 * vb[i] + (1 - b2) * gb[i] * gb[i];
            }
        }
    }

    // m_ = m / (1 - b1_t)
    // v_ = v / (1 - b2_t)
    // w -= learning_rate * m_ / (sqrt(v_) + sigma)
    // b -= learning_rate * m_ / (sqrt(v_) + sigma)
    void update_model(Model *model, GradW const &grads_w, GradB const &grads_b,
                      double learning_rate) {
        for (size_t l = 0; l < grads_w.size(); ++l) {
            Matrix w = model->layers[l].weights;

            for (size_t i = 0; i < w.rows * w.cols; ++i) {
                double mm = m_w[l].mem[i] / (1 - b1_t);
                double vv = v_w[l].mem[i] / (1 - b2_t);
                w.mem[i] -= learning_rate * mm / (std::sqrt(vv) + sigma);
            }
        }

        for (size_t l = 0; l < grads_b.size(); ++l) {
            Vector b = model->layers[l].biases;

            for (size_t i = 0; i < b.size; ++i) {
                double mm = m_b[l].mem[i] / (1 - b1_t);
                double vv = v_b[l].mem[i] / (1 - b2_t);
                b.mem[i] -= learning_rate * mm / (std::sqrt(vv) + sigma);
            }
        }
    }

    void execute(Model *model, GradW const &grads_w, GradB const &grads_b,
                 double learning_rate) override {
        if (is_init == false) [[unlikely]] {
            init(grads_w, grads_b);
        }
        compute_mv(grads_w, grads_b);
        update_model(model, grads_w, grads_b, learning_rate);
        b1_t *= b1;
        b2_t *= b2;
    }

    bool is_init = false;
    double b1 = 0.9;
    double b2 = 0.999;
    double b1_t = 0.9;
    double b2_t = 0.999;
    double sigma = 1e-8;
    std::vector<Matrix> m_w;
    std::vector<Vector> m_b;
    std::vector<Matrix> v_w;
    std::vector<Vector> v_b;
};

/******************************************************************************/
/*                                 functions                                  */
/******************************************************************************/

Vector map(ActivationFunction *act, Vector const &v);
Vector map_derivative(ActivationFunction *act, Vector const &v);
Vector map(CostFunction *cost, Vector const &v1, Vector const &v2);
Vector map_derivative(CostFunction *cost, Vector const &v1, Vector const &v2);

#endif
