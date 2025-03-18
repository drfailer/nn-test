#ifndef TYPES_H
#define TYPES_H
#include "matrix.hpp"
#include <vector>

using Vectors = std::vector<Vector>;

struct DataSetEntry {
    Vector input;
    Vector ground_truth;
};

using DataSet = std::vector<DataSetEntry>;

using GradW = std::vector<Matrix>;
using GradB = std::vector<Vector>;

#endif
