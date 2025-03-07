#ifndef MINIBATCH_GENERATOR_H
#define MINIBATCH_GENERATOR_H
#include "types.hpp"
#include <cstdint>
#include <random>

class MinibatchGenerator {
  public:
    MinibatchGenerator(size_t size, uint32_t seed)
        : size_(size), indexes_(size), gen_(seed) {
        for (size_t i = 0; i < size; ++i) {
            indexes_[i] = i;
        }
    }

    void generate(DataBase const &db) {
        this->db_ = &db;
        std::uniform_int_distribution<size_t> dist(0, size_);
        std::random_shuffle(indexes_.begin(), indexes_.end());
    }

    DataBaseEntry const &get(size_t idx) const {
        return (*db_)[indexes_[idx]];
    }

    size_t size() const { return size_; }

  private:
    DataBase const *db_ = nullptr;
    size_t size_ = 0;
    std::vector<size_t> indexes_;
    std::mt19937 gen_;
};

#endif
