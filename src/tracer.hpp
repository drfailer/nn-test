#ifndef TRACER_H
#define TRACER_H
#include "trainer.hpp"
#include "types.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

struct Tracer {
    std::vector<ftype> costs_train = {};
    std::vector<ftype> costs_test = {};
    std::vector<ftype> accuracy_train = {};
    std::vector<ftype> accuracy_test = {};
    size_t nb_epochs = 0;
    size_t minibatch_size = 0;
    ftype learning_rate = 0;
    DataSet const &train_ds = {};
    DataSet const &test_ds = {};
    size_t loading_count = 0;

    Tracer(DataSet const &train_ds, DataSet const &test_ds)
        : train_ds(train_ds), test_ds(test_ds) {}

    void init(size_t nb_epochs, size_t minibatch_size, ftype learning_rate) {
        this->nb_epochs = nb_epochs;
        this->minibatch_size = minibatch_size;
        this->learning_rate = learning_rate;
        this->costs_train = std::vector<ftype>(nb_epochs);
        this->costs_test = std::vector<ftype>(nb_epochs);
        this->accuracy_train = std::vector<ftype>(nb_epochs);
        this->accuracy_test = std::vector<ftype>(nb_epochs);
        this->loading_count = std::max<size_t>(1, nb_epochs / 100);
    }

    void trace(Trainer const *trainer, size_t epoch) {
        auto eval_train = trainer->evaluate(train_ds);
        auto eval_test = trainer->evaluate(test_ds);
        costs_train[epoch] = eval_train.first;
        costs_test[epoch] = eval_test.first;
        accuracy_train[epoch] = eval_train.second;
        accuracy_test[epoch] = eval_test.second;
        if (epoch % loading_count == 0 || epoch == nb_epochs) {
            std::cout << "trace " << 100 * epoch / nb_epochs << " %"
                      << std::endl;
        }
    }

    void dump() {
        std::ostringstream ss;
        ss << "train_" << nb_epochs << "_" << learning_rate << "_" <<
            minibatch_size << ".out";

        std::ofstream fs(ss.str());

        fs.write(reinterpret_cast<char const *>(&nb_epochs), sizeof(size_t));
        fs.write(reinterpret_cast<char const *>(&minibatch_size),
                 sizeof(size_t));
        fs.write(reinterpret_cast<char const *>(&learning_rate),
                 sizeof(ftype));
        fs.write(reinterpret_cast<char const *>(costs_train.data()),
                 nb_epochs * sizeof(ftype));
        fs.write(reinterpret_cast<char const *>(accuracy_train.data()),
                 nb_epochs * sizeof(ftype));
        fs.write(reinterpret_cast<char const *>(costs_test.data()),
                 nb_epochs * sizeof(ftype));
        fs.write(reinterpret_cast<char const *>(accuracy_test.data()),
                 nb_epochs * sizeof(ftype));
    }
};

#endif
