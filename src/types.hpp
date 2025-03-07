#ifndef TYPES_H
#define TYPES_H
#include "matrix.hpp"
#include <vector>

using Vectors = std::vector<Vector>;

struct DataBaseEntry {
    Vector input;
    Vector ground_truth;
};

using DataBase = std::vector<DataBaseEntry>;

using GradW = std::vector<Matrix>;
using GradB = std::vector<Vector>;

#endif
