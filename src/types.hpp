#ifndef TYPES_H
#define TYPES_H
#include "matrix.hpp"
#include <vector>

using Vectors = std::vector<Vector>;
using DataBase = std::vector<std::pair<Vector, Vector>>;

using GradW = std::vector<Matrix>;
using GradB = std::vector<Vector>;

#endif
