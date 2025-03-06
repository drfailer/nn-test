#ifndef MATRIX_H
#define MATRIX_H
#include <cstddef>
#include <cstring>
#include <functional>
#include <initializer_list>

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
        memcpy(this->mem, m.mem, rows * cols * sizeof(*mem));
    }
    Matrix const &operator=(Matrix const &m) {
        if (&m == this) return *this;
        delete[] this->mem;
        this->mem = new double[m.rows * m.cols];
        this->rows = m.rows;
        this->cols = m.cols;
        memcpy(this->mem, m.mem, rows * cols * sizeof(*mem));
        return *this;
    }

    Matrix(Matrix &&m) : mem(nullptr), rows(m.rows), cols(m.cols) {
        std::swap(this->mem, m.mem);
    }
    Matrix const &operator=(Matrix &&m) {
        this->rows = m.rows;
        this->cols = m.cols;
        std::swap(this->mem, m.mem);
        return *this;
    }

    ~Matrix() {
        delete[] this->mem;
        this->mem = nullptr;
        this->rows = 0;
        this->cols = 0;
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
        std::swap(this->mem, v.mem);
    }
    Vector const &operator=(Vector &&v) {
        this->size = v.size;
        std::swap(this->mem, v.mem);
        return *this;
    }

    Vector(Vector const &v): Vector(v.size) {
        memcpy(mem, v.mem, size * sizeof(*mem));
    }
    Vector const &operator=(Vector const &v) {
        if (&v == this) return *this;
        delete[] this->mem;
        this->mem = new double[v.size];
        this->size = v.size;
        return *this;
    }

    ~Vector() {
        delete[] this->mem;
        this->mem = nullptr;
        this->size = 0;
    }

    Vector clone() const {
        Vector result(size);
        memcpy(result.mem, mem, size * sizeof(*mem));
        return result;
    }

    double &operator[](size_t idx) { return mem[idx]; }
    double const &operator[](size_t idx) const { return mem[idx]; }
};

/******************************************************************************/
/*                                 functions                                  */
/******************************************************************************/

Vector map(std::function<double(double)> fun, Vector const &v);
Vector map(std::function<double(double, double)> fun, Vector const &v1,
           Vector const &v2);
Vector matmul(Matrix const &weights, Vector &err);
Matrix matmul(Vector const &err, Vector const &a);
Vector hadamard(Vector &&a, Vector const &b);

/******************************************************************************/
/*                                 operators                                  */
/******************************************************************************/

struct LazyCVMult {
    double c;
    Vector const &v;
};

LazyCVMult operator*(double constant, Vector const &vector);
Vector const &operator-=(Vector &v, LazyCVMult const &cvmult);

struct LazyCMMult {
    double c;
    Matrix const &m;
};

LazyCMMult operator*(double constant, Matrix const &m);
Matrix const &operator-=(Matrix &m, LazyCMMult const &cvmult);

#endif
