#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import shutil

import numpy as np
import tensorflow as tf


INPUT_SIZE = 50
N_TRAIN_SAMPLES = 10240
# just random guess of 16 lstm cells
HIDDEN_UNITS = 16
BATCH_SIZE = 256
LEARNING_RATE = 0.05

TRAINING_STEPS = 1000
DISPLAY_STEP = TRAINING_STEPS / 100

TENSORBOARD_DIR = "./tensorboard"


def generate_dataset(n=N_TRAIN_SAMPLES):
    max_50_bit_number = 2**50 - 1
    random_numbers = np.random.randint(0, max_50_bit_number, n)
    X_strings = list(map(lambda x: "{:050b}".format(x), random_numbers))
    # print(X_strings)
    y = []
    for x_s in X_strings:
        if x_s.count('1') % 2 == 0:
            y.append([1,0])
        else:
            y.append([0,1])

    X = [[int(i) for i in s] for s in X_strings]
    return (X, y)


def build_and_train_model(dataset):
    print("[..] Building network model")

    # input tensor X is #{BATCH_SIZE} of 50bits lists
    X = tf.placeholder(tf.float32, (BATCH_SIZE, INPUT_SIZE), name="X")
    # the answer is either even [1, 0] or odd [0, 1] 
    y = tf.placeholder(tf.float32, (BATCH_SIZE, 2), name="y")

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_UNITS)

    # cast inputs to the shape `static_rnn` wants: list of 50 elements every of size [BATCH_SIZE, 1]
    inputs = tf.reshape(X, [BATCH_SIZE, INPUT_SIZE, 1])
    inputs = tf.unstack(inputs, INPUT_SIZE, axis=1)
    assert(len(inputs) == INPUT_SIZE)
    assert(inputs[0].shape == (BATCH_SIZE, 1) )
    outputs, final_state = tf.nn.static_rnn(lstm_cell, inputs, dtype=tf.float32)

    with tf.name_scope("Logits"):
        Why = tf.get_variable("Why", shape=[HIDDEN_UNITS, 2],
                              initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=[2,],
                            initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.matmul(outputs[-1], Why) + b

    with tf.name_scope("CE"):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
        tf.summary.scalar("CE_loss", loss_op)

    with tf.name_scope("accuracy"):
        predictions = tf.nn.softmax(logits)
        assert(tf.argmax(predictions, axis=1).shape == tf.argmax(y, axis=1).shape)
        correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(y, axis=1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar("accuracy", accuracy_op)

    with tf.name_scope("train"):
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_op)

    summ = tf.summary.merge_all()

    print("[..] Training the model")
    X_train, y_train = np.array(dataset[0]), np.array(dataset[1])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        writer = tf.summary.FileWriter(TENSORBOARD_DIR)
        writer.add_graph(sess.graph)

        for step in range(TRAINING_STEPS):
            i = 0
            for batch_start in range(0, N_TRAIN_SAMPLES, BATCH_SIZE):
                batch_end = batch_start + BATCH_SIZE
                _, loss_val, acc_val = sess.run([train_op, loss_op, accuracy_op],
                                       feed_dict= {X: X_train[batch_start : batch_end],
                                                   y: y_train[batch_start : batch_end]})

                i += 1
                if i % 5 == 0:
                    s = sess.run(summ, feed_dict={X: X_train[batch_start : batch_end],
                                                  y: y_train[batch_start : batch_end]})
                    writer.add_summary(s, step)

            if (step+1) % DISPLAY_STEP == 0:
                # writer = tf.summary.FileWriter('.')
                # writer.add_graph(tf.get_default_graph())
                print("Step {0}, loss: {1:.4}, accuracy: {2:.4}".format(step, loss_val, acc_val))


def clean_logs_from_previous_run():
    print("[..] Removing \"{}\" dir".format(TENSORBOARD_DIR))
    shutil.rmtree(TENSORBOARD_DIR, ignore_errors=True)


def main():
    clean_logs_from_previous_run()
    dataset = generate_dataset(N_TRAIN_SAMPLES)
    # print(dataset)
    build_and_train_model(dataset)


if __name__ == '__main__':
    main()
