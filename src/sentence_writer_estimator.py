# using TensorFlows custom Estimator API

import numpy as np
import tensorflow as tf

import random
import csv
import itertools
import nltk
#nltk.download('punkt')
import pickle

import datetime


VOCAB_SIZE = 7
NUM_LAYERS = 1
NUM_NEURONS = 10
LEARNING_RATE = 0.01
SAVE_PATH = "./saves/"
FILE_TRAIN = "test_sentences.txt"

# only one feature column right now
NUM_TIMESTEPS = 1


def my_input_fn(file_path, repeat_count=1, shuffle_count=1):

    # first some pre processing

    # SENTENCE_START i like to code . SENTENCE_END




    # then return the batch_features and batch_labels
    # dictionary's keys are the names of the features, and the dictionary's values are the feature's values
    batch_features = {'word0': [0, 1, 2, 3, 4]}

    # which is a list of the label's values for a batch
    batch_labels = [1, 2, 3, 4, 5]

    print(batch_features)
    print(batch_labels)

    return batch_features, batch_labels


word0 = tf.feature_column.categorical_column_with_identity(
    key='word0',
    num_buckets=1,
)

# feature_columns = [tf.reshape(tf.feature_column.embedding_column(word0, VOCAB_SIZE), [-1, VOCAB_SIZE, 1])]
feature_columns = [tf.feature_column.embedding_column(word0, VOCAB_SIZE)]




def my_model_fn(
        features,   # =batch_features
        labels,     # =batch_labels
        mode        # instance of tf.estimator.ModeKeys
):
    """
    Function must:
        -create architecture of model
        -define behavior in TRAIN, EVAL, PREDICT modes
    """

    input_layer = tf.feature_column.input_layer(features, feature_columns)
    print(input_layer)
    # creating architecture
    # X_placeholder = tf.placeholder(tf.float32, [None, NUM_TIMESTEPS, VOCAB_SIZE])

    # need to take labels and create the y_placeholder equivalent
    #y_placeholder = tf.placeholder(tf.float32, [None, NUM_TIMESTEPS, VOCAB_SIZE])
    correct_words = tf.one_hot(labels, VOCAB_SIZE, on_value=1.0, off_value=0.0)

    def single_cell():
        return tf.contrib.rnn.BasicLSTMCell(num_units=NUM_NEURONS, activation=tf.nn.relu)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [single_cell() for _ in range(NUM_LAYERS)])

    lstm_with_wrapper = tf.contrib.rnn.OutputProjectionWrapper(
        stacked_lstm,
        output_size=VOCAB_SIZE)

    init_state = single_cell().zero_state(1, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_with_wrapper, tf.reshape(input_layer, shape=[1, VOCAB_SIZE, 1]), dtype=tf.float32, initial_state=None)


    # learning_rate = tf.placeholder(tf.float32, shape=[])
    # learning_rate = tf.cast(LEARNING_RATE, tf.float32)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=correct_words))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train = optimizer.minimize(loss)

    # sentence_loss_pl = tf.placeholder(tf.float32, [])

    # init = tf.global_variables_initializer()

    # my_checkpoint_config = tf.estimator.RunConfig(keep_checkpoint_max=2)

    # return init, train, loss, X_placeholder, y_placeholder, outputs, sentence_loss_pl

    loss_summary = tf.summary.scalar('Loss', loss)

    predictions = outputs

    # if using the model to predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        print("predicting")
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # if evaluating the model
    elif mode == tf.estimator.ModeKeys.EVAL:
        print("evaluating")
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'loss': loss_summary})

    # training the model
    else:
        print("training")
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train)





model = tf.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir=SAVE_PATH
)

model.train(
    input_fn=lambda: my_input_fn(FILE_TRAIN)
)




