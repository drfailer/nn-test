#ifndef MATH_H
#define MATH_H
#include "cblas.h"
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <vector>

using ftype = float;

/******************************************************************************/
/*                                   types                                    */
/******************************************************************************/

struct Matrix {
    ftype *mem = nullptr;
    size_t rows = 0;
    size_t cols = 0;

    Matrix() = default;
    Matrix(size_t rows, size_t cols)
        : mem(new ftype[rows * cols]), rows(rows), cols(cols) {}

    Matrix(Matrix const &m) : Matrix(m.rows, m.cols) {
        memcpy(mem, m.mem, rows * cols * sizeof(*mem));
    }
    Matrix const &operator=(Matrix const &m) {
        if (&m == this)
            return *this;
        if (rows * cols != m.rows * m.cols) {
            delete[] mem;
            mem = new ftype[m.rows * m.cols];
        }
        rows = m.rows;
        cols = m.cols;
        memcpy(mem, m.mem, rows * cols * sizeof(*mem));
        return *this;
    }

    Matrix(Matrix &&m) : mem(nullptr), rows(m.rows), cols(m.cols) {
        std::swap(mem, m.mem);
    }
    Matrix const &operator=(Matrix &&m) {
        rows = m.rows;
        cols = m.cols;
        std::swap(mem, m.mem);
        return *this;
    }

    ~Matrix() {
        delete[] mem;
        mem = nullptr;
        rows = 0;
        cols = 0;
    }

    ftype *operator[](size_t idx) { return &mem[idx * cols]; }
    ftype const *operator[](size_t idx) const { return &mem[idx * cols]; }
};

struct Vector {
    ftype *mem = nullptr;
    size_t size = 0;

    Vector() = default;
    explicit Vector(size_t size) : mem(new ftype[size]), size(size) {}
    Vector(std::initializer_list<ftype> init) : Vector(init.size()) {
        memcpy(mem, std::data(init), init.size() * sizeof(*mem));
    }

    Vector(Vector &&v) : mem(nullptr), size(v.size) { std::swap(mem, v.mem); }
    Vector const &operator=(Vector &&v) {
        size = v.size;
        std::swap(mem, v.mem);
        return *this;
    }

    Vector(Vector const &v) : Vector(v.size) {
        memcpy(mem, v.mem, size * sizeof(*mem));
    }
    Vector const &operator=(Vector const &v) {
        if (&v == this)
            return *this;
        if (size != v.size) {
            delete[] mem;
            mem = new ftype[v.size];
        }
        size = v.size;
        memcpy(mem, v.mem, size * sizeof(*mem));
        return *this;
    }

    ~Vector() {
        delete[] mem;
        mem = nullptr;
        size = 0;
    }

    Vector clone() const {
        Vector result(size);
        memcpy(result.mem, mem, size * sizeof(*mem));
        return result;
    }

    ftype &operator[](size_t idx) { return mem[idx]; }
    ftype const &operator[](size_t idx) const { return mem[idx]; }
};

template <typename MatrixType>
    requires std::is_same_v<MatrixType, Matrix> ||
             std::is_same_v<MatrixType, Vector>
struct T {
    MatrixType const &matrix;
    explicit T(MatrixType const &matrix) : matrix(matrix) {}
};

using GradW = std::vector<Matrix>;
using GradB = std::vector<Vector>;

/******************************************************************************/
/*                                 functions                                  */
/******************************************************************************/

Vector matmul(T<Matrix> const &weightsT, Vector &err);
Matrix matmul(Vector const &err, T<Vector> const &aT);
Vector hadamard(Vector &&a, Vector const &b);

/******************************************************************************/
/*                                 operators                                  */
/******************************************************************************/

struct LazyCVMult {
    ftype c;
    Vector const &v;
};

LazyCVMult operator*(ftype constant, Vector const &v);
Vector const &operator-=(Vector &v, LazyCVMult const &cvmult);
Vector const &operator+=(Vector &lhs, Vector const &rhs);
Vector const &operator/=(Vector &v, ftype constant);

struct LazyCMMult {
    ftype c;
    Matrix const &m;
};

LazyCMMult operator*(ftype constant, Matrix const &m);
Matrix const &operator-=(Matrix &m, LazyCMMult const &cvmult);
Matrix const &operator+=(Matrix &lhs, Matrix const &rhs);
Matrix const &operator/=(Matrix &m, ftype constant);

GradW const &operator+=(GradW &lhs, GradW const &rhs);
GradB const &operator+=(GradB &lhs, GradB const &rhs);

/******************************************************************************/
/*                              helper for cblas                              */
/******************************************************************************/

template <typename T>
void gemv(CBLAS_TRANSPOSE const trans, int const m, int const n, T const alpha,
          T const *a, int const lda, T const *x, int const incx, T const beta,
          T *y, int const incy) {
    if constexpr (std::is_same_v<ftype, double>) {
        cblas_dgemv(CblasRowMajor, trans, m, n, alpha, a, lda, x, incx, beta, y,
                    incy);
    } else {
        cblas_sgemv(CblasRowMajor, trans, m, n, alpha, a, lda, x, incx, beta, y,
                    incy);
    }
}

template <typename T>
void gemm(CBLAS_TRANSPOSE const TransA, CBLAS_TRANSPOSE const TransB,
          int const M, int const N, int const K, T const alpha, T const *A,
          int const lda, T const *B, int const ldb, T const beta, T *C,
          int const ldc) {
    if constexpr (std::is_same_v<ftype, double>) {
        cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, ldc);
    } else {
        cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, ldc);
    }
}

#endif
