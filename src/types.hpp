#ifndef TYPES_H
#define TYPES_H
#include "math.hpp"
#include <vector>

using Vectors = std::vector<Vector>;

struct DataSetEntry {
    Vector input;
    Vector ground_truth;
};

using DataSet = std::vector<DataSetEntry>;

#endif
