#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import tensorflow as tf


# 50 bits long string
INPUT_SIZE = 50
N_TRAIN_SAMPLES = 100000
# hidden state of LSTM cell equal to sequence length
HIDDEN_UNITS = 50
BATCH_SIZE = 1000
LEARNING_RATE = 0.005

TRAINING_STEPS = 3000
DISPLAY_STEP = TRAINING_STEPS / 100


def generate_dataset(n=N_TRAIN_SAMPLES):
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


def build_and_train_model(dataset):
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
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
    tf.summary.scalar("CE_loss", loss_op)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss_op)

    summ = tf.summary.merge_all()

    print("[..] Training the model")
    X_train, y_train = np.array(dataset[0]), np.array(dataset[1])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        writer = tf.summary.FileWriter("./tensorboard")
        writer.add_graph(sess.graph)

        for step in range(TRAINING_STEPS):
            i = 0
            for batch_start in range(0, N_TRAIN_SAMPLES, BATCH_SIZE):
                loss_val, _ = sess.run([loss_op, train_op],
                                       {X: X_train[batch_start : batch_start + BATCH_SIZE],
                                        y: y_train[batch_start : batch_start + BATCH_SIZE]})

                i += 1
                if i % 5 == 0:
                    s = sess.run(summ, feed_dict={X: X_train[batch_start : batch_start + BATCH_SIZE],
                                                  y: y_train[batch_start : batch_start + BATCH_SIZE]})
                    writer.add_summary(s, step)

            if (step+1) % DISPLAY_STEP == 0:
                # writer = tf.summary.FileWriter('.')
                # writer.add_graph(tf.get_default_graph())
                print("Step {0}, loss: {1}, accuracy:".format(step, loss_val))


def main():
    dataset = generate_dataset(N_TRAIN_SAMPLES)
    train_op = build_and_train_model(dataset)


if __name__ == '__main__':
    main()
