#ifndef MATH_H
#define MATH_H
#include <cstddef>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <type_traits>

/******************************************************************************/
/*                                   types                                    */
/******************************************************************************/

struct Matrix {
    double *mem = nullptr;
    size_t rows = 0;
    size_t cols = 0;

    Matrix() = default;
    Matrix(size_t rows, size_t cols)
        : mem(new double[rows * cols]), rows(rows), cols(cols) {}

    Matrix(Matrix const &m) : Matrix(m.rows, m.cols) {
        memcpy(mem, m.mem, rows * cols * sizeof(*mem));
    }
    Matrix const &operator=(Matrix const &m) {
        if (&m == this) return *this;
        if (rows * cols != m.rows * m.cols) {
            delete[] mem;
            mem = new double[m.rows * m.cols];
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

    double *operator[](size_t idx) { return &mem[idx * cols]; }
    double const *operator[](size_t idx) const { return &mem[idx * cols]; }
};

struct Vector {
    double *mem = nullptr;
    size_t size = 0;

    Vector() = default;
    explicit Vector(size_t size) : mem(new double[size]), size(size) {}
    Vector(std::initializer_list<double> init) : Vector(init.size()) {
        memcpy(mem, std::data(init), init.size() * sizeof(*mem));
    }

    Vector(Vector &&v) : mem(nullptr), size(v.size) {
        std::swap(mem, v.mem);
    }
    Vector const &operator=(Vector &&v) {
        size = v.size;
        std::swap(mem, v.mem);
        return *this;
    }

    Vector(Vector const &v): Vector(v.size) {
        memcpy(mem, v.mem, size * sizeof(*mem));
    }
    Vector const &operator=(Vector const &v) {
        if (&v == this) return *this;
        if (size != v.size) {
            delete[] mem;
            mem = new double[v.size];
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

    double &operator[](size_t idx) { return mem[idx]; }
    double const &operator[](size_t idx) const { return mem[idx]; }
};

template <typename MatrixType>
    requires std::is_same_v<MatrixType, Matrix> ||
             std::is_same_v<MatrixType, Vector>
struct T {
    MatrixType const &matrix;
    explicit T(MatrixType const &matrix): matrix(matrix) {}
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
    double c;
    Vector const &v;
};

LazyCVMult operator*(double constant, Vector const &v);
Vector const &operator-=(Vector &v, LazyCVMult const &cvmult);
Vector const &operator+=(Vector &lhs, Vector const &rhs);
Vector const &operator/=(Vector &v, double constant);

struct LazyCMMult {
    double c;
    Matrix const &m;
};

LazyCMMult operator*(double constant, Matrix const &m);
Matrix const &operator-=(Matrix &m, LazyCMMult const &cvmult);
Matrix const &operator+=(Matrix &lhs, Matrix const &rhs);
Matrix const &operator/=(Matrix &m, double constant);

GradW const &operator+=(GradW &lhs, GradW const &rhs);
GradB const &operator+=(GradB &lhs, GradB const &rhs);

#endif
