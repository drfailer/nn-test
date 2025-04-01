#include "math.hpp"
#include <cassert>
#include <cblas.h>
#include <cstring>

/******************************************************************************/
/*                                 functions                                  */
/******************************************************************************/

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
    gemm<ftype>(CblasNoTrans, CblasTrans, err.size, a.size, 1, 1.0, err.mem, 1,
                a.mem, 1, 0, result.mem, result.cols);
    return result;
}

Vector matmul(T<Matrix> const &weightsT, Vector &err) {
    Matrix const &weights = weightsT.matrix;
    Vector result(weights.cols);

    assert(weights.rows == err.size);
    gemv<ftype>(CblasTrans, weights.rows, weights.cols, 1.0, weights.mem,
                weights.cols, err.mem, 1, 0, result.mem, 1);
    return result;
}

/******************************************************************************/
/*                                 operators                                  */
/******************************************************************************/


LazyCVMult operator*(ftype constant, Vector const &v) {
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

Vector const &operator/=(Vector &v, ftype constant) {
    for (size_t i = 0; i < v.size; ++i) {
        v.mem[i] /= constant;
    }
    return v;
}

LazyCMMult operator*(ftype constant, Matrix const &m) {
    return LazyCMMult(constant, m);
}

Matrix const &operator-=(Matrix &m, LazyCMMult const &cvmult) {
    for (size_t i = 0; i < (m.rows * m.cols); ++i) {
        m.mem[i] -= cvmult.c * cvmult.m.mem[i];
    }
    return m;
}

Matrix const &operator+=(Matrix &lhs, Matrix const &rhs) {
    for (size_t i = 0; i < (lhs.rows * rhs.cols); ++i) {
        lhs.mem[i] += rhs.mem[i];
    }
    return lhs;
}

Matrix const &operator/=(Matrix &m, ftype constant) {
    for (size_t i = 0; i < (m.rows * m.cols); ++i) {
        m.mem[i] /= constant;
    }
    return m;
}

GradW const &operator+=(GradW &lhs, GradW const &rhs) {
    assert(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] += rhs[i];
    }
    return lhs;
}

GradB const &operator+=(GradB &lhs, GradB const &rhs) {
    assert(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] += rhs[i];
    }
    return lhs;
}
