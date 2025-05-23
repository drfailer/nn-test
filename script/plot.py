#!/usr/bin/env python3

import matplotlib.pyplot as plt
import struct
import argparse


class Parser:
    def __init__(self):
        self.nb_epochs = 0
        self.minibatch_size = 0
        self.learning_rate = 0
        self.costs_train = []
        self.accuracy_train = []
        self.costs_test = []
        self.accuracy_test = []

    def parse_file(self, filename):
        with open(filename, "rb") as file:
            content = file.read()
            self.nb_epochs, self.minibatch_size, self.learning_rate =\
                struct.unpack("<QQf", content[:8 + 8 + 4])

            print(self.nb_epochs)
            print(self.minibatch_size)
            print(self.learning_rate)
            print(len(content[8 + 8 + 4:]))
            it = struct.iter_unpack("<f", content[8 + 8 + 4:])

            self.costs_train = [next(it) for _ in range(self.nb_epochs)]
            self.accuracy_train = [next(it) for _ in range(self.nb_epochs)]
            self.costs_test = [next(it) for _ in range(self.nb_epochs)]
            self.accuracy_test = [next(it) for _ in range(self.nb_epochs)]


def plot(filename):
    fig, ax = plt.subplots(2, 1, squeeze=False)
    parser = Parser()

    parser.parse_file(filename)

    ax[0, 0].set_title("Evolution of the cost per epochs")
    ax[0, 0].plot(parser.costs_train, label="train")
    ax[0, 0].plot(parser.costs_test, label="test")
    ax[0, 0].set_xlabel("epochs")
    ax[0, 0].set_ylabel("cost")
    ax[0, 0].legend()

    ax[1, 0].set_title("Evolution of the accuracy per epochs")
    ax[1, 0].plot(parser.accuracy_train, label="train")
    ax[1, 0].plot(parser.accuracy_test, label="test")
    ax[1, 0].set_xlabel("epochs")
    ax[1, 0].set_ylabel("accuracy (%)")
    ax[1, 0].legend()

    fig.suptitle(f"epochs = {parser.nb_epochs}, minibatch_size = {parser.minibatch_size}, learning_rate = {parser.learning_rate}")
    plt.show()


def plot_adam_sgd(files):
    fig, ax = plt.subplots(2, 1, squeeze=False)

    ax[0, 0].set_title("Evolution of the cost per epochs")
    ax[0, 0].set_xlabel("epochs")
    ax[0, 0].set_ylabel("cost")
    ax[0, 0].legend()

    ax[1, 0].set_title("Evolution of the accuracy per epochs")
    ax[1, 0].set_xlabel("epochs")
    ax[1, 0].set_ylabel("accuracy (%)")
    ax[1, 0].legend()

    for file in files:
        parser = Parser()
        elements = file.split(':')
        parser.parse_file(elements[0])
        label = elements[-1]

        ax[0, 0].plot(parser.costs_test, label=label)
        ax[1, 0].plot(parser.accuracy_test, label=label)
        fig.suptitle(f"epochs = {parser.nb_epochs}, minibatch_size = {parser.minibatch_size}, learning_rate = {parser.learning_rate}")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser("plot")
    parser.add_argument("files", nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.files) > 1:
        plot_adam_sgd(args.files)
    else:
        plot(args.files[0])


if __name__ == "__main__":
    main()
