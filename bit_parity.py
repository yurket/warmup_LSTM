#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import tensorflow as tf


# 50 bits long string
INPUT_SIZE = 50
# hidden state of LSTM cell equal to sequence length
HIDDEN_UNITS = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.001

def generate_dataset(n=200):
    max_50_bit_number = 2**50 - 1
    random_numbers = np.random.randint(0, max_50_bit_number, n)
    X_strings = map(lambda x: "{:050b}".format(x), random_numbers)
    y = []
    for x_s in X_strings:
        if x_s.count('1') % 2 == 0:
            y.append([1,0])
        else:
            y.append([0,1])

    X = [[int(i) for i in s] for s in X_strings]
    return (X, y)

def create_rnn_cell(cell_type='rnn'):
    if cell_type == 'rnn':
        return tf.nn.rnn_cell.BasicRNNCell(num_units=HIDDEN_UNITS)
    elif cell_type == 'gru':
        return tf.nn.rnn_cell.GRUCell(num_units=HIDDEN_UNITS)
    elif cell_type == 'lstm':
        return tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_UNITS)


def build_model(dataset):
    print("[..] Building network model")
    X = tf.placeholder(tf.float32, (BATCH_SIZE, INPUT_SIZE), name="X")
    y = tf.placeholder(tf.float32, (BATCH_SIZE, 2), name="y")

    rnn_cell = create_rnn_cell('rnn')
    state = tf.zeros([BATCH_SIZE, HIDDEN_UNITS])
    _, state = rnn_cell(X, state)

    Why = tf.get_variable("Why", shape=[HIDDEN_UNITS, 2],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=[2,],
        initializer=tf.contrib.layers.xavier_initializer())

    logits = tf.matmul(state, Why) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss)


    print("[..] Training the model")
    X_train, y_train = np.array(dataset[0]), np.array(dataset[1])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(train_op, {X: X_train, y: y_train})


def main():
    dataset = generate_dataset()
    print("Dataset: ")
    print("X: ", dataset[0])
    print("y: ", dataset[1])
    train_op = build_model(dataset)


if __name__ == '__main__':
    main()
