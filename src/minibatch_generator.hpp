#ifndef MINIBATCH_GENERATOR_H
#define MINIBATCH_GENERATOR_H
#include "types.hpp"
#include <algorithm>
#include <cstdint>
#include <random>

class MinibatchGenerator {
  public:
    MinibatchGenerator(DataSet const &db, size_t size, uint32_t seed)
        : dataSet_(&db), size_(size), indexes_(db.size()), gen_(seed),
          offset_(0) {
        for (size_t i = 0; i < db.size(); ++i) {
            indexes_[i] = i;
        }
        std::shuffle(indexes_.begin(), indexes_.end(), gen_);
    }

    void generate() {
        if (offset_ + size_ >= dataSet_->size()) {
            offset_ = 0;
            std::shuffle(indexes_.begin(), indexes_.end(), gen_);
        } else {
            offset_ += size_;
        }
    }

    DataSetEntry const &get(size_t idx) const {
        return (*dataSet_)[indexes_[offset_ + idx]];
    }

    size_t size() const { return size_; }

  private:
    DataSet const *dataSet_ = nullptr;
    size_t size_ = 0;
    std::vector<size_t> indexes_;
    std::mt19937 gen_;
    size_t offset_ = 0;
};

#endif
