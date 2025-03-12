#include "matrix.hpp"
#include <cassert>
#include <cblas.h>
#include <cstring>

/******************************************************************************/
/*                                 functions                                  */
/******************************************************************************/

Vector map(std::function<double(double)> fun, Vector const &v) {
    Vector result(v.size);

    for (size_t i = 0; i < v.size; ++i) {
        result[i] = fun(v[i]);
    }
    return result;
}

Vector map(std::function<double(double, double)> fun, Vector const &v1,
                    Vector const &v2) {
    Vector result(v1.size);

    assert(v1.size == v2.size);
    for (size_t i = 0; i < v1.size; ++i) {
        result[i] = fun(v1[i], v2[i]);
    }
    return result;
}

Vector hadamard(Vector &&a, Vector const &b) {
    assert(a.size == b.size);
    for (size_t i = 0; i < a.size; ++i) {
        a[i] *= b[i];
    }
    return a;
}

Matrix matmul(Vector const &err, T<Vector> const &aT) {
    Vector const &a = aT.matrix;
    Matrix result(err.size, a.size);

    // A:err -> err.sizex1 (MxK)
    // B:aT -> 1xa.size (KxN)
    // M = err.size, K = 1, N = a.size
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, err.size, a.size, 1,
                1.0, err.mem, 1, a.mem, 1, 0, result.mem, result.cols);
    return result;
}

Vector matmul(T<Matrix> const &weightsT, Vector &err) {
    Matrix const &weights = weightsT.matrix;
    Vector result(weights.cols);

    assert(weights.rows == err.size);
    cblas_dgemv(CblasRowMajor, CblasTrans, weights.rows, weights.cols, 1.0,
                weights.mem, weights.cols, err.mem, 1, 0, result.mem, 1);
    return result;
}

/******************************************************************************/
/*                                 operators                                  */
/******************************************************************************/


LazyCVMult operator*(double constant, Vector const &v) {
    return LazyCVMult(constant, v);
}

Vector const &operator-=(Vector &v, LazyCVMult const &cvmult) {
    for (size_t i = 0; i < v.size; ++i) {
        v.mem[i] -= cvmult.c * cvmult.v.mem[i];
    }
    return v;
}

Vector const &operator+=(Vector &lhs, Vector const &rhs) {
    for (size_t i = 0; i < lhs.size; ++i) {
        lhs.mem[i] += rhs.mem[i];
    }
    return lhs;
}

Vector const &operator/=(Vector &v, double constant) {
    for (size_t i = 0; i < v.size; ++i) {
        v.mem[i] /= constant;
    }
    return v;
}

LazyCMMult operator*(double constant, Matrix const &m) {
    return LazyCMMult(constant, m);
}

Matrix const &operator-=(Matrix &m, LazyCMMult const &cvmult) {
    for (size_t i = 0; i < (m.rows * m.cols); ++i) {
        m.mem[i] -= cvmult.c * cvmult.m.mem[i];
    }
    return m;
}

Matrix const &operator+=(Matrix &lhs, Matrix const &rhs) {
    for (size_t i = 0; i < (lhs.rows * rhs.rows); ++i) {
        lhs.mem[i] += rhs.mem[i];
    }
    return lhs;
}

Matrix const &operator/=(Matrix &m, double constant) {
    for (size_t i = 0; i < (m.rows * m.rows); ++i) {
        m.mem[i] /= constant;
    }
    return m;
}
