#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include "math.hpp"
#include "model.hpp"
#include <cassert>
#include <cmath>

/******************************************************************************/
/*                                 interfaces                                 */
/******************************************************************************/

/* struct CostFunction { */
/*     virtual ftype execute(ftype ground_truth, ftype layer_output) = 0; */
/*     virtual ftype derivative(ftype ground_truth, ftype layer_output) = 0; */
/* }; */

/* struct ActivationFunction { */
/*     virtual ftype execute(ftype) = 0; */
/*     virtual ftype derivative(ftype) = 0; */
/* }; */

/* struct OptimizeFunction { */
/*     virtual void execute(Model *model, GradW const &grads_w, */
/*                          GradB const &grads_b, ftype learning_rate) = 0; */
/* }; */

/******************************************************************************/
/*                              implementations                               */
/******************************************************************************/

struct QuadraticLoss {
    ftype execute(ftype ground_truth, ftype output) const {
        ftype diff = ground_truth - output;
        return 0.5 * diff * diff;
    }

    ftype derivative(ftype ground_truth, ftype output) const {
        return output - ground_truth;
    }
};

struct Sigmoid {
    ftype execute(ftype x) const { return 1.0 / (1.0 + std::exp(-x)); }

    ftype derivative(ftype x) const {
        return execute(x) * (1.0 - execute(x));
    }
};

struct SGD {
    void execute(Model *model, GradW const &grads_w, GradB const &grads_b,
                 ftype learning_rate) {
        for (size_t l = 0; l < model->layers.size(); ++l) {
            assert(grads_w[l].rows == model->layers[l].nb_nodes &&
                   grads_w[l].cols == model->layers[l].nb_inputs);
            assert(grads_b[l].size == model->layers[l].nb_nodes);

            model->layers[l].weights -= learning_rate * grads_w[l];
            model->layers[l].biases -= learning_rate * grads_b[l];
        }
    }
};

struct Adam {
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
            ftype *mw = m_w[l].mem;
            ftype *vw = v_w[l].mem;
            ftype const *gw = grads_w[l].mem;

            for (size_t i = 0; i < grads_w[l].rows * grads_w[l].cols; ++i) {
                mw[i] = b1 * mw[i] + (1 - b1) * gw[i];
                vw[i] = b2 * vw[i] + (1 - b2) * gw[i] * gw[i];
            }
        }

        for (size_t l = 0; l < grads_b.size(); ++l) {
            ftype *mb = m_b[l].mem;
            ftype *vb = v_b[l].mem;
            ftype const *gb = grads_b[l].mem;

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
                      ftype learning_rate) {
        for (size_t l = 0; l < grads_w.size(); ++l) {
            ftype *w = model->layers[l].weights.mem;
            ftype const *mw = m_w[l].mem;
            ftype const *vw = v_w[l].mem;

            for (size_t i = 0; i < grads_w[l].rows * grads_w[l].cols; ++i) {
                ftype mm = mw[i] / (1 - b1_t);
                ftype vv = vw[i] / (1 - b2_t);
                w[i] -= learning_rate * mm / (std::sqrt(vv) + sigma);
            }
        }

        for (size_t l = 0; l < grads_b.size(); ++l) {
            ftype *b = model->layers[l].biases.mem;
            ftype const *mb = m_b[l].mem;
            ftype const *vb = v_b[l].mem;

            for (size_t i = 0; i < grads_b[l].size; ++i) {
                ftype mm = mb[i] / (1 - b1_t);
                ftype vv = vb[i] / (1 - b2_t);
                b[i] -= learning_rate * mm / (std::sqrt(vv) + sigma);
            }
        }
    }

    void execute(Model *model, GradW const &grads_w, GradB const &grads_b,
                 ftype learning_rate) {
        if (is_init == false) [[unlikely]] {
            init(grads_w, grads_b);
        }
        compute_mv(grads_w, grads_b);
        update_model(model, grads_w, grads_b, learning_rate);
        b1_t *= b1;
        b2_t *= b2;
    }

    bool is_init = false;
    ftype b1 = 0.9;
    ftype b2 = 0.999;
    ftype b1_t = 0.9;
    ftype b2_t = 0.999;
    ftype sigma = 1e-8;
    std::vector<Matrix> m_w;
    std::vector<Vector> m_b;
    std::vector<Matrix> v_w;
    std::vector<Vector> v_b;
};

/******************************************************************************/
/*                                 functions                                  */
/******************************************************************************/

template <typename Activation>
Vector map(Activation const &act, Vector const &v) {
    Vector result(v.size);

    for (size_t i = 0; i < v.size; ++i) {
        result[i] = act.execute(v[i]);
    }
    return result;
}

template <typename Activation>
Vector map_derivative(Activation const &act, Vector const &v) {
    Vector result(v.size);

    for (size_t i = 0; i < v.size; ++i) {
        result[i] = act.derivative(v[i]);
    }
    return result;
}

template <typename Cost>
Vector map(Cost const &cost, Vector const &v1, Vector const &v2) {
    Vector result(v1.size);

    assert(v1.size == v2.size);
    for (size_t i = 0; i < v1.size; ++i) {
        result[i] = cost.execute(v1[i], v2[i]);
    }
    return result;
}

template <typename Cost>
Vector map_derivative(Cost const &cost, Vector const &v1, Vector const &v2) {
    Vector result(v1.size);

    assert(v1.size == v2.size);
    for (size_t i = 0; i < v1.size; ++i) {
        result[i] = cost.derivative(v1[i], v2[i]);
    }
    return result;
}

#endif
