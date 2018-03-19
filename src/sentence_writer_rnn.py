### rnn that can generate sentences

import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

import random
import csv
import itertools
import nltk
#nltk.download('punkt')
import pickle

import time
import datetime

#to start tensorboard using conda:
#python /anaconda3/envs/conda_TensorSpeak/lib/python3.6/site-packages/tensorflow/tensorboard/tensorboard.py --logdir=./summary_log/

unknown_token = "unknown_token"
sentence_start_token = "sentence_start"
sentence_end_token = "sentence_end"

class Model(object):
    def __init__(self, num_io, num_timesteps, num_layers, num_neurons_inlayer, learning_rate, num_iterations, batch_size):
        self.num_io = num_io
        self.num_timesteps = num_timesteps
        self.num_layers = num_layers
        self.num_neurons_inlayer = num_neurons_inlayer
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.vocab_size = num_io
        self.corpus_file_name = ""
        self.current_save_name = ""

        # variables initialized later
        self.word_to_index = []
        self.index_to_word = []


    def print_model_info(self):
        print("Model info:")
        print("num_io: %d" % self.num_io)
        print("num_timesteps: %d" % self.num_timesteps)
        print("num_layers: %d" % self.num_layers)
        print("num_neurons_inlayer: %d" % self.num_neurons_inlayer)
        print("learning_rate: %.5f" % self.learning_rate)
        print("num_iterations: %d" % self.num_iterations)
        print("batch_size: %d" % self.batch_size)
        print("vocab_size: %d" % self.vocab_size)
        print("corpus_file_name: %s" % self.corpus_file_name)
        print("current_save_name: %s" % self.current_save_name)


    def build_graph(self):
        print("Building model")
        #run tensorflow functions to build model

        X_placeholder = tf.placeholder(tf.float32, [None, self.num_timesteps, self.num_io])
        y_placeholder = tf.placeholder(tf.float32, [None, self.num_timesteps, self.num_io])

        def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(num_units=self.num_neurons_inlayer, activation=tf.nn.relu)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [single_cell() for _ in range(self.num_layers)])

        lstm_with_wrapper = tf.contrib.rnn.OutputProjectionWrapper(
            stacked_lstm,
            output_size=self.num_io)

        outputs, states = tf.nn.dynamic_rnn(lstm_with_wrapper, X_placeholder, dtype=tf.float32)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y_placeholder))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        return init, train, loss, X_placeholder, y_placeholder, outputs

    def train_model(self, vectorized_sentences):
        print("Training started at: " + datetime.datetime.now().strftime("%H:%M:%S"))

        init, train, loss, X_placeholder, y_placeholder, outputs = self.build_graph()

        saver = tf.train.Saver()
        loss_summary = tf.summary.scalar('Loss', loss)

        ### TRAINING MODEL
        with tf.Session() as sess:
            sess.run(init)

            #TensorBoard code
            summaryMerged = tf.summary.merge_all()
            self.update_save_name()
            filename = "../logs/" + self.current_save_name
            writer = tf.summary.FileWriter(filename, sess.graph)
            tensorboard_counter = 0


            for iteration in range(self.num_iterations):
                print("BEGINNING ITERATION #" + str(iteration) + "\tTime: " + datetime.datetime.now().strftime("%H:%M:%S"))
                mse_count = 0
                avg_mse = 0
                np.random.shuffle(vectorized_sentences)

                sentence_count = 0
                for sent_index, sentence in enumerate(vectorized_sentences):

                    for word in range(len(sentence)-self.num_timesteps):
                        X_batch = [sentence[word:word+self.num_timesteps]]
                        y_batch = [sentence[word+1:word+1+self.num_timesteps]]
                        #print("X_batch:\n" + str(X_batch))
                        #print("y_batch:\n" + str(y_batch) + '\n')

                        loss_result, train_result = sess.run([loss_summary, train], feed_dict={X_placeholder: X_batch, y_placeholder: y_batch})

                        writer.add_summary(loss_result, tensorboard_counter)
                        tensorboard_counter = tensorboard_counter + 1


                    if sentence_count % 10 == 0:
                        print("sentence_count: %d" % sentence_count)
                    sentence_count = sentence_count + 1

            variables_save_file = "../models/" + self.corpus_file_name + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
            saver.save(sess,  variables_save_file)
            writer.close()
        #self.graph_mse()


    def generate_sentences(self, graph_name, starting_sentence):
        print("generating sentences")

        init, train, loss, X_placeholder, y_placeholder, outputs = self.build_graph()

        saver = tf.train.Saver()

        print(self.word_to_index)

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
                #print("Feeding: " + str(generated_sentence[x:x+self.num_timesteps,:].reshape(1, self.num_timesteps, self.vocab_size)))
                sess.run(tf.nn.softmax(logits=outputs), feed_dict={X_placeholder: generated_sentence[x:x+self.num_timesteps,:].reshape(1, self.num_timesteps, self.vocab_size)})

            while (sentence_end_token != self.index_to_word[generated_sentence[-1,:].argmax()]) & (curr_sentence_length < max_sentence_length):
                # feed in current word, predict next word
                #print("generated_sentence.shape: " + str(generated_sentence.shape))

                curr_words_vector = generated_sentence[-self.num_timesteps:, :].reshape(1,-1, self.vocab_size)
                pred_words_vector = sess.run(tf.nn.softmax(logits=outputs), feed_dict={X_placeholder: curr_words_vector})

                #print("Fed: " + str(curr_words_vector))
                #print("Pred: " + str(pred_words_vector))

                #print("pred_words_vector.shape " + str(pred_words_vector.shape))

                generated_sentence = np.append(generated_sentence, pred_words_vector[:, -1, :]).reshape( -1, self.vocab_size)
                #print("generated_sentence.shape " + str(generated_sentence.shape))

                #curr_words_vector = generated_sentence[:, -self.num_timesteps:, :]
                #print("curr_words_vector.shape " + str(curr_words_vector.shape))


                pred_word_str = self.index_to_word[generated_sentence[0, -1].argmax()]
                fed_word_str = self.index_to_word[generated_sentence[0, -2].argmax()]
                #print("Word #" + str(curr_sentence_length))
                #print("Fed: " + fed_word_str)
                #print("Predicted: " + pred_word_str)

                #generated_sentence.append(self.index_to_word[pred_words_vector.argmax()])

                curr_sentence_length = curr_sentence_length + 1


            translated_sentence = ""
            for word in generated_sentence:
                translated_sentence = translated_sentence + " " + self.index_to_word[word.argmax()]


            print('\t' + '\t' + '\t' + '\t' + '\t' + translated_sentence)


    def load_sentences(self, corpus_file_name, max_sent_len = 70):

        self.corpus_file_name = corpus_file_name
        self.current_save_name = self.corpus_file_name + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
        ### tokenize sentences, call create_dictionary

        print("Reading CSV file: %s" % corpus_file_name)

        # create string of start and end tokens to buffer either side of each sentence
        sentence_start_string = ""
        sentence_end_string = ""
        #testing with only 1 start token, then buffering with zero arrays
        #for x in range(self.num_timesteps):
        for x in range(self.num_timesteps):
            sentence_start_string = sentence_start_string + sentence_start_token + " "
            sentence_end_string = sentence_end_string + " " + sentence_end_token

        with open("../data/" + corpus_file_name + ".csv", 'rt') as f:
            reader = csv.reader(f, skipinitialspace=True)
            #reader.next()

            # read all csv lines and filter out empty lines
            csv_lines = [x for x in reader]
            csv_lines_filtered = filter(None, csv_lines)

            # tokenize sentences and attach start/end tokens
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in csv_lines_filtered])
            sentences = ["%s %s %s" % (sentence_start_string, sent, sentence_end_string) for sent in sentences]

        #print(sentences)

        # tokenize sentences into words using TweetTokenizer to preserve handles
        tk = nltk.TweetTokenizer(strip_handles=False, reduce_len=False, preserve_case=False)
        tokenized_sentences = [tk.tokenize(sent) for sent in sentences]

        # find max sentence length
        max_sent_rec = 0
        for i, sent in enumerate(tokenized_sentences):
            if len(tokenized_sentences[i]) > max_sent_rec:
                max_sent_rec = len(tokenized_sentences[i])
        print("Longest sentence is %d words" % (max_sent_rec))


        if max_sent_len > 0:
            # get rid of sentences longer than max_sent_len, optional argument
            total_num_sentences_untrimmed = len(tokenized_sentences)
            tokenized_sentences = [sent for sent in tokenized_sentences if len(sent) <= (max_sent_len)]
            print("%d out of %d sentences are %d-words-long or less." % (
            len(tokenized_sentences), total_num_sentences_untrimmed, max_sent_len))

        # create dictionary of words
        self.create_dictionary(corpus_file_name, tokenized_sentences)

        # replace all words not in our vocabulary with the unknown token
        # add start and end sentence tokens
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in self.word_to_index else unknown_token for w in sent]

        #print(tokenized_sentences)
        return tokenized_sentences


    def create_dictionary(self, corpus_file_name, tokenized_sentences):
        print("creating dictionary")

        # create dictionary of words
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words." % len(word_freq.items()))
        vocab = word_freq.most_common(self.vocab_size - 1)
        self.index_to_word = [x[0] for x in vocab]
        self.index_to_word.append(unknown_token)
        self.word_to_index = dict([w, i] for i, w in enumerate(self.index_to_word))

        print("Using a vocab of %d words." % self.vocab_size)

        # save index_to_word and word_to_index as pkls
        with open("../models/" + self.current_save_name + "_INDEX_TO_WORD_" + '.pkl', 'wb') as f:
            pickle.dump(self.index_to_word, f, pickle.HIGHEST_PROTOCOL)
        with open("../models/" + self.current_save_name + "_WORD_TO_INDEX" + '.pkl', 'wb') as f:
            pickle.dump(self.word_to_index, f, pickle.HIGHEST_PROTOCOL)


    def load_dictionary(self, dictionary_name):
        print("loading dictionary from: %s" % self.current_save_name)

        with open("../models/" + dictionary_name + "_INDEX_TO_WORD_" + '.pkl', 'rb') as f:
            self.index_to_word = pickle.load(f)
        with open("../models/" + dictionary_name + "_WORD_TO_INDEX" + '.pkl', 'rb') as f:
            self.word_to_index = pickle.load(f)

    def sentences_to_vectors(self, tokenized_sentences):

        vectorized_sentences = []

        for sentence in tokenized_sentences:
            vectorized_sentence = np.array([])
            for word_str in sentence:
                word_index = self.word_to_index[word_str]
                vectorized_word = np.zeros(self.vocab_size).tolist()
                vectorized_word[word_index] = 1

                vectorized_sentence = np.append(vectorized_sentence, vectorized_word).reshape(-1, self.vocab_size).tolist()


            vectorized_sentences.append(vectorized_sentence)

        return vectorized_sentences

    #BACKUP
    """
        def sentences_to_vectors(self, tokenized_sentences):

        vectorized_sentences = []

        for sentence in tokenized_sentences:
            vectorized_sentence = []
            for word_str in sentence:
                word_index = self.word_to_index[word_str]
                vectorized_word = np.zeros(self.vocab_size).tolist()
                vectorized_word[word_index] = 1

                vectorized_sentence.append(vectorized_word)

            vectorized_sentences.append(vectorized_sentence)

        return vectorized_sentences
    
    """


    def update_save_name(self):
        self.current_save_name = self.corpus_file_name + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")

def save_model_object(model):
    print("Saving model:")
    # save model as a .pkl

    model.update_save_name()
    with open("../models/" + model.current_save_name + '.pkl', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def load_model_object(filename):
    print("Loading model: %s" % filename)
    # load model as a .pkl

    with open("../models/" + filename + ".pkl", 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

# TRAINING
def train_it():

    model = Model(num_io=4000, num_timesteps=3, num_layers=5, num_neurons_inlayer=50,
                       learning_rate=0.001, num_iterations=1, batch_size=1)

    tokenized_sentences = model.load_sentences("trump_10k_tweets")
    save_model_object(model)
    model.print_model_info()

    vectorized_sentences = model.sentences_to_vectors(tokenized_sentences)
    model.train_model(vectorized_sentences)




# GENERATING
def generate_it():
    dictionary_name = "trump_100_tweets_03-18--01-43"
    model_name = "trump_100_tweets_03-18--01-43"
    graph_name = "trump_100_tweets_03-18--05-59"

    model = load_model_object(model_name)
    model.print_model_info()

    model.load_dictionary(dictionary_name)
    model.generate_sentences(graph_name, "the")


def check_model(model_name):
    model = load_model_object(model_name)
    model.print_model_info()

train_it()
#generate_it()

#check_model("test_sentences_03-18--00-41")


