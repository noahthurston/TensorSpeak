"""
RNN that can generate sentences.
Written by Noah Thurston
"""

import numpy as np
import tensorflow as tf
import csv
import itertools
import nltk
#nltk.download('punkt')
import pickle
import datetime


unknown_token = "unknown_token"
sentence_start_token = "sentence_start"
sentence_end_token = "sentence_end"

class Model(object):
    """
    The Model class stores attributes about the RNN and the data it's trained on.
    Has methods to load, build, and train the tf graph.
    """
    def __init__(self, corpus_file_name, num_io, num_timesteps, num_layers, num_neurons_inlayer, learning_rate, batch_size, max_sentence_len=30):
        self.num_io = num_io
        self.num_timesteps = num_timesteps
        self.num_layers = num_layers
        self.num_neurons_inlayer = num_neurons_inlayer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vocab_size = num_io
        self.corpus_file_name = corpus_file_name
        self.current_save_name = ""
        self.max_sentence_len = max_sentence_len

        # variables initialized later
        self.word_to_index = []
        self.index_to_word = []
        self.indexed_sentences = []

    def print_model_info(self):
        print("\n")
        print("Model info:")
        print("num_io: %d" % self.num_io)
        print("num_timesteps: %d" % self.num_timesteps)
        print("num_layers: %d" % self.num_layers)
        print("num_neurons_inlayer: %d" % self.num_neurons_inlayer)
        print("learning_rate: %.8f" % self.learning_rate)
        print("batch_size: %d" % self.batch_size)
        print("vocab_size: %d" % self.vocab_size)
        print("corpus_file_name: %s" % self.corpus_file_name)
        print("current_save_name: %s" % self.current_save_name)
        print("\n")

    def build_graph(self):
        print("Building model")
        #run tensorflow functions to build model

        X_placeholder = tf.placeholder(tf.float32, [None, self.num_timesteps, self.num_io])
        y_placeholder = tf.placeholder(tf.float32, [None, self.num_timesteps, self.num_io])

        # with tf.variable_scope('rnn', initializer=tf.variance_scaling_initializer()):
        def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(num_units=self.num_neurons_inlayer, activation=tf.nn.elu)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [single_cell() for _ in range(self.num_layers)])

        lstm_with_wrapper = tf.contrib.rnn.OutputProjectionWrapper(
            stacked_lstm,
            output_size=self.num_io)

        outputs, states = tf.nn.dynamic_rnn(lstm_with_wrapper, X_placeholder, dtype=tf.float32)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y_placeholder))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)

        sentence_loss_pl = tf.placeholder(tf.float32, [])

        init = tf.global_variables_initializer()

        return init, train, loss, X_placeholder, y_placeholder, outputs, sentence_loss_pl

    def train_model(self, num_sentences_to_train, save_every=20000, graph_name=""):
        print("Training started at: " + datetime.datetime.now().strftime("%H:%M:%S"))
        init, train, loss, X_placeholder, y_placeholder, outputs, sentence_loss_pl = self.build_graph()

        saver = tf.train.Saver(max_to_keep=100)
        loss_summary = tf.summary.scalar('Loss', loss)

        sentence_loss_list = np.array([])

        sentence_loss_summary = tf.summary.scalar('Sentence Loss', sentence_loss_pl)


        ### TRAINING MODEL
        with tf.Session() as sess:
            sess.run(init)

            if graph_name != "":
                print("Loading graph: %s" % ("../models/" + graph_name))
                saver.restore(sess, "../models/" + graph_name)

            #TensorBoard code
            #summaryMerged = tf.summary.merge_all()
            self.update_save_name()
            filename = "../logs/" + self.current_save_name
            writer = tf.summary.FileWriter(filename, sess.graph)
            tensorboard_word_counter = 0
            tensorboard_sentence_counter = 0

            curr_sentence_in_training = 0

            learning_rate = tf.placeholder(tf.float32, shape=[])

            while curr_sentence_in_training < num_sentences_to_train:
                print("\nITERATION #%d\tTime: %s" % (curr_sentence_in_training/len(self.indexed_sentences), datetime.datetime.now().strftime("%H:%M:%S")))
                # saving due to beginning of iteration over training set
                variables_save_file = "../models/" + self.corpus_file_name + "_" + datetime.datetime.now().strftime(
                    "%m-%d--%H-%M")
                saver.save(sess, variables_save_file)
                print("\nTrained %d sentences\tTime: %s" % (
                curr_sentence_in_training, datetime.datetime.now().strftime("%H:%M:%S")))
                print("Saved graph to: %s" % variables_save_file)
                self.save()

                np.random.shuffle(self.indexed_sentences)

                sent_index = 0
                while (sent_index < len(self.indexed_sentences)) & (curr_sentence_in_training < num_sentences_to_train):
                    vectorized_sentence = self.indexed_sentence_to_vectors(self.indexed_sentences[sent_index])
                    sentence_loss_list = np.array([])
                    for word in range(len(vectorized_sentence)-self.num_timesteps):
                        X_batch = [vectorized_sentence[word:word+self.num_timesteps]]
                        y_batch = [vectorized_sentence[word+1:word+1+self.num_timesteps]]

                        loss_result, train_result, word_loss = sess.run([loss_summary, train, loss], feed_dict={X_placeholder: X_batch, y_placeholder: y_batch, learning_rate: self.learning_rate})
                        sentence_loss_list = np.append(sentence_loss_list, word_loss)


                        # add tensorboard summary for the curr word MSE
                        writer.add_summary(loss_result, tensorboard_word_counter)
                        tensorboard_word_counter = tensorboard_word_counter + 1

                    sentence_loss_result_evaluated = np.mean(sentence_loss_list)

                    sentence_loss_result = sess.run(sentence_loss_summary, feed_dict={sentence_loss_pl:sentence_loss_result_evaluated})
                    writer.add_summary(sentence_loss_result, tensorboard_sentence_counter)
                    tensorboard_sentence_counter = tensorboard_sentence_counter + 1

                    if curr_sentence_in_training % 2500 == 0:
                        print(curr_sentence_in_training)

                    if curr_sentence_in_training % save_every == 0:
                        # saving due to the "save_every" condition
                        variables_save_file = "../models/" + self.corpus_file_name + "_" + datetime.datetime.now().strftime(
                            "%m-%d--%H-%M")
                        saver.save(sess, variables_save_file)
                        print("\nTrained %d sentences\tTime: %s" % (curr_sentence_in_training, datetime.datetime.now().strftime("%H:%M:%S")))
                        print("Saved graph to: %s" % variables_save_file)
                        self.save()
                    sent_index = sent_index + 1
                    curr_sentence_in_training = curr_sentence_in_training + 1

            # save once done training
            print("FINISHED TRAINING, NOW SAVING")
            variables_save_file = "../models/" + self.corpus_file_name + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
            print("\nTrained %d sentences\tTime: %s" % (curr_sentence_in_training, datetime.datetime.now().strftime("%H:%M:%S")))
            print("Saved graph to: %s" % variables_save_file)
            saver.save(sess, variables_save_file)
            writer.close()
            self.save()
        return

    def sample_word_from_softmax(self, prev_sent, pred_sent, temperature=0):
        pred_word = pred_sent[0, -1, :]

        if temperature != 0:
            # if temperature is non-zero, sample normally from dist
            # print("with temp")
            pred_word = self.soft_with_temp(pred_word, temp=temperature)
        else:
            # else just pick the most popular word
            # print("just pick most popular")
            pred_word = pred_sent[0, -1, :]

        max_to_sample = 1000
        num_sampled = 0
        curr_index = self.word_to_index[unknown_token]

        while (curr_index == self.word_to_index[unknown_token]) & (
                num_sampled < max_to_sample):
            # print("sampling")
            random_num = np.random.uniform(0, 1, 1)
            curr_index = 0
            curr_sum = pred_word[curr_index]
            while curr_sum < random_num:
                curr_index += 1
                curr_sum += pred_word[curr_index]
            num_sampled += 1

        # check if it maxed-out sampling, just pick second largest argmax
        if num_sampled == max_to_sample:
            curr_index = pred_word.argsort[-2]

        new_word = np.zeros(self.vocab_size)
        new_word[curr_index] = 1

        extended_sentence = np.append(prev_sent, new_word).reshape(-1, self.vocab_size)

        return extended_sentence

    def soft_with_temp(self, distribution, temp=0.025):
        new_distribution = np.zeros(len(distribution))

        sum = 0
        for val in distribution:
            sum += np.exp(val / temp)

        for index in range(len(new_distribution)):
            new_distribution[index] = np.exp(distribution[index] / temp) / sum

        # print(new_distribution)
        return new_distribution

    def generate_sentences(self, graph_name, starting_sentence, temperature):
        print("generating sentences")

        init, train, loss, X_placeholder, y_placeholder, outputs, sentence_loss_pl = self.build_graph()

        saver = tf.train.Saver()

        print("starting_sentence: " + str(starting_sentence))

        tk = nltk.TweetTokenizer(strip_handles=False, reduce_len=False, preserve_case=False)
        tokenized_sentence = tk.tokenize(starting_sentence)

        # replace all words not in our vocabulary with the unknown token
        # add start and end sentence tokens
        for i, word in enumerate(tokenized_sentence):
            tokenized_sentence[i] = word if word in self.word_to_index else unknown_token

        sentence_start_vector = np.zeros(self.vocab_size)
        sentence_start_vector[self.word_to_index[sentence_start_token]] = 1

        generated_sentence = np.array([])
        for x in range(self.num_timesteps):
            generated_sentence = np.append(generated_sentence, sentence_start_vector).reshape(-1, self.vocab_size)

        for word_str in tokenized_sentence:
            word_index = self.word_to_index[word_str]
            vectorized_word = np.zeros(self.vocab_size)
            vectorized_word[word_index] = 1

            generated_sentence = np.append(generated_sentence, vectorized_word).reshape(-1, self.vocab_size)

        ### GENERATING SENTENCES WITH MODEL
        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, "../models/" + graph_name)

            max_sentence_length = 40
            curr_sentence_length = len(generated_sentence) - self.num_timesteps

            #feed sentence seed through model
            for x in range(len(generated_sentence) - self.num_timesteps):
                sess.run(tf.nn.softmax(logits=outputs), feed_dict={X_placeholder: generated_sentence[x:x+self.num_timesteps,:].reshape(1, self.num_timesteps, self.vocab_size)})

            while (sentence_end_token != self.index_to_word[generated_sentence[-1,:].argmax()]) & (curr_sentence_length < max_sentence_length):
                # feed in current word, predict next word

                curr_words_vector = generated_sentence[-self.num_timesteps:, :].reshape(1,-1, self.vocab_size)
                pred_words_vector = sess.run(tf.nn.softmax(logits=outputs), feed_dict={X_placeholder: curr_words_vector})

                generated_sentence = self.sample_word_from_softmax(generated_sentence, pred_words_vector, temperature=temperature)

                curr_sentence_length = curr_sentence_length + 1

            translated_sentence = ""
            for word in generated_sentence:
                translated_sentence = translated_sentence + " " + self.index_to_word[word.argmax()]

        return translated_sentence

    def indexed_sentence_to_vectors(self, sentence):
        # this function receives in a sentence in the form (0, 0, 4, 3, 7, 1, 1, 1)
        # outputs a list of 1 hot vectors
        vectorized_sentence = np.array([])

        for word in sentence:
            vectorized_word = np.zeros(self.vocab_size)
            vectorized_word[int(word)] = 1
            vectorized_sentence = np.append(vectorized_sentence, vectorized_word).reshape(-1, self.vocab_size)

        return vectorized_sentence

    def save(self):
        self.update_save_name()
        print("Saving model: %s" % self.current_save_name)
        # save model as a .pkl

        with open("../models/" + self.current_save_name + '.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        print("Loading model: %s" % filename)
        # load model as a .pkl

        with open("../models/" + filename + ".pkl", 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model

    def update_save_name(self):
        self.current_save_name = self.corpus_file_name + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
