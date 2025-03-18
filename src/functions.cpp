#include "functions.hpp"
#include "math.hpp"
#include <cassert>

Vector map(ActivationFunction *act, Vector const &v) {
    Vector result(v.size);

    for (size_t i = 0; i < v.size; ++i) {
        result[i] = act->execute(v[i]);
    }
    return result;
}

Vector map_derivative(ActivationFunction *act, Vector const &v) {
    Vector result(v.size);

    for (size_t i = 0; i < v.size; ++i) {
        result[i] = act->derivative(v[i]);
    }
    return result;
}

Vector map(CostFunction *cost, Vector const &v1, Vector const &v2) {
    Vector result(v1.size);

    assert(v1.size == v2.size);
    for (size_t i = 0; i < v1.size; ++i) {
        result[i] = cost->execute(v1[i], v2[i]);
    }
    return result;
}

Vector map_derivative(CostFunction *cost, Vector const &v1, Vector const &v2) {
    Vector result(v1.size);

    assert(v1.size == v2.size);
    for (size_t i = 0; i < v1.size; ++i) {
        result[i] = cost->derivative(v1[i], v2[i]);
    }
    return result;
}
