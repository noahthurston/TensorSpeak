### rnn that can generate sentences

import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

import csv
import itertools
import nltk
#nltk.download('punkt')
import pickle

# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler


unknown_token = "unknown_token"
sentence_start_token = "sentence_start"
sentence_end_token = "sentence_end"


class Model(object):
    def __init__(self, num_io, num_timesteps, num_neurons_inlayer, learning_rate, num_iterations, batch_size, save_dir):
        self.num_io = num_io
        self.num_timesteps = num_timesteps
        self.num_neurons_inlayer = num_neurons_inlayer
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.vocab_size = num_io

        # variables initialized later
        # self.vectorized_sentences = []
        self.word_to_index = []
        self.index_to_word = []

    def train_model(self, vectorized_sentences):
        print("training")

        ### BUILDING MODEL
        X_placeholder = tf.placeholder(tf.float32, [None, self.num_timesteps, self.num_io])
        y_placeholder = tf.placeholder(tf.float32, [None, self.num_timesteps, self.num_io])

        cell = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.BasicLSTMCell(num_units=self.num_neurons_inlayer, activation=tf.nn.relu),
            output_size=self.num_io)

        outputs, states = tf.nn.dynamic_rnn(cell, X_placeholder, dtype=tf.float32)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y_placeholder))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


        ### TRAINING MODEL
        with tf.Session() as sess:
            sess.run(init)

            for sent_index, sentence in enumerate(vectorized_sentences):
                # iter_div = int(iteration / 100)
                #print("New sentence")

                if sent_index%100 == 0:
                    print("Sentence: %d", sent_index)

                for word in range(len(sentence)-1):
                    X_batch = [[sentence[word]]]
                    y_batch = [[sentence[word+1]]]
                    #print("X_batch:")
                    #print(X_batch)
                    #print("y_batch:")
                    #print(y_batch)

                    sess.run(train, feed_dict={X_placeholder: X_batch, y_placeholder: y_batch})


                    #curr_mse = loss.eval(feed_dict={X_placeholder: X_batch, y_placeholder: y_batch})
                    #print("Current MSE:  %f" % curr_mse)

            saver.save(sess, self.save_dir + "saved_model")


    def generate_sentences(self, num_sentences):
        print("generating sentences")

        ### BUILDING MODEL
        X_placeholder = tf.placeholder(tf.float32, [None, self.num_timesteps, self.num_io])
        y_placeholder = tf.placeholder(tf.float32, [None, self.num_timesteps, self.num_io])

        cell = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.BasicLSTMCell(num_units=self.num_neurons_inlayer, activation=tf.nn.relu),
            output_size=self.num_io)

        outputs, states = tf.nn.dynamic_rnn(cell, X_placeholder, dtype=tf.float32)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y_placeholder))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


        ### GENERATING SENTENCES WITH MODEL
        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, self.save_dir + "saved_model")


            for generated_sent_index in range(num_sentences):
                curr_word_vector = np.array([[[1 if x == self.word_to_index["!"] else 0 for x in range(self.vocab_size)]]])

                max_sentence_length = 30
                curr_sentence_length = 0

                print("Feeding: %s" % self.index_to_word[curr_word_vector.argmax()])

                generated_sentence = []

                while (sentence_end_token != self.index_to_word[curr_word_vector.argmax()]) & (curr_sentence_length < max_sentence_length):
                    # feed in current word, predict next word
                    pred_word_vector = sess.run(tf.nn.softmax(logits=outputs), feed_dict={X_placeholder: curr_word_vector})
                    curr_word_vector = pred_word_vector

                    pred_word_str = self.index_to_word[curr_word_vector.argmax()]
                    print("Predicted: " + pred_word_str)
                    print("Feeding: %s" % self.index_to_word[pred_word_vector.argmax()])

                    generated_sentence.append(self.index_to_word[pred_word_vector.argmax()])

                    curr_sentence_length + curr_sentence_length + 1

                print('\t' + '\t' + '\t' + '\t' + '\t' + str(generated_sentence[:-1]))



        #### target index = myList.index(max(myList))



    #def get_next_batch(self, vectorized_sentences, iteration):


    def load_sentences(self, corpus_file_name, max_sent_len = -1):

        ### tokenize sentences, call create_dictionary

        print("Reading CSV file: %s" % corpus_file_name)

        with open(self.save_dir + corpus_file_name + ".csv", 'rt') as f:
            reader = csv.reader(f, skipinitialspace=True)
            #reader.next()

            # read all csv lines and filter out empty lines
            csv_lines = [x for x in reader]
            csv_lines_filtered = filter(None, csv_lines)

            # tokenize sentences and attach start/end tokens
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in csv_lines_filtered])
            sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

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
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in self.word_to_index else unknown_token for w in sent]

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

        # save index_to_word and word_to_index as pckls
        with open(self.save_dir + "INDEX_TO_WORD_" + corpus_file_name + '.pkl', 'wb') as f:
            pickle.dump(self.index_to_word, f, pickle.HIGHEST_PROTOCOL)
        with open(self.save_dir +  "WORD_TO_INDEX" + corpus_file_name + '.pkl', 'wb') as f:
            pickle.dump(self.word_to_index, f, pickle.HIGHEST_PROTOCOL)


    def load_dictionary(self, corpus_file_name):
        print("loading dictionary")

        with open(self.save_dir + "INDEX_TO_WORD_" + corpus_file_name + '.pkl', 'rb') as f:
            self.index_to_word = pickle.load(f)
        with open(self.save_dir + "WORD_TO_INDEX" + corpus_file_name + '.pkl', 'rb') as f:
            self.word_to_index = pickle.load(f)

    def sentences_to_vectors(self, tokenized_sentences):
        vectorized_sentences = []
        for sentence in tokenized_sentences:
            vectorized_sentence = []
            for word_index in sentence:
                vectorized_word = [1 if x == self.word_to_index[sentence_start_token] else 0 for x in range(self.vocab_size)]
                vectorized_sentence.append(vectorized_word)
            vectorized_sentences.append(vectorized_sentence)

        return vectorized_sentences

# def __init__(self, num_io, num_timesteps, num_neurons_inlayer, learning_rate, num_iterations, batch_size, save_dir):



test_model = Model(num_io=8000, num_timesteps=1, num_neurons_inlayer=10,
                   learning_rate=0.03, num_iterations=3, batch_size=1, save_dir="../data/")

#def load_sentences(self, corpus_path, max_sent_len = -1):


tokenized_sentences = test_model.load_sentences("test_sentences")
"""
print("tokenize_sentences:")
print(tokenized_sentences)
print('\n')

print("index_to_word:")
print(test_model.index_to_word)
print('\n')

print("word_to_index:")
print(test_model.word_to_index)
print('\n')
"""

# turn each sentence into a list of vectors of length vocab_size

#print(vectorized_sentences[0])
#print(vectorized_sentences[1])
#print(vectorized_sentences[2])

#print(vectorized_sentences)


# vectorized_sentences = test_model.sentences_to_vectors(tokenized_sentences)
#test_model.train_model(vectorized_sentences)
test_model.load_dictionary("test_sentences")
test_model.generate_sentences(3)




"""
each sentence looks like:
[
  [0 0 0 1]
  [0 0 1 0]
  [0 1 0 0]
  [1 0 0 0]
  [0 1 0 0]
  [0 0 1 0]
]

were going to feed in words like this
[[
  [0 0 0 1]
]]
"""



###    TO DO    ###
"""
-start runnable-code practice model
-check what tokenized sentences looks like (shape)
-write function to vectorize sentences so that each word is one-hot-vector
    (MUST BE ABLE TO FEED INTO MODEL) 

-DECIDE:
    -feed single words, predict next single word (MODEL #1)
        or
    -feed ~10 words in a row, predict next timestep forward (MODEL #2)

"""








raise SystemExit


#### old code
"""
# ##################################################################################################
# ##################################################################################################


MyModelDesc = {"num_io":4,
           "num_timesteps":6,
           "num_neurons_inlayer":100,
           "learning_rate":0.030,
           "num_iterations":1000,
           "batch_size":1}

X_placeholder = tf.placeholder(tf.float32, [None, MyModelDesc["num_timesteps"], MyModelDesc["num_io"]])
y_placeholder = tf.placeholder(tf.float32, [None, MyModelDesc["num_timesteps"], MyModelDesc["num_io"]])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=MyModelDesc["num_neurons_inlayer"], activation=tf.nn.relu),output_size=MyModelDesc["num_io"])

outputs, states = tf.nn.dynamic_rnn(cell, X_placeholder, dtype=tf.float32)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=y_placeholder))
optimizer = tf.train.AdamOptimizer(learning_rate=MyModelDesc["learning_rate"])
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#    X_set = full_set[:,:-1].reshape(-1, steps, 1)
#    y_set = full_set[:,1:].reshape(-1, steps, 1)


def get_next_batch(ModelDesc):
    test_arr = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    test_arr_x5 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]

    num_test_vectors = len(test_arr_x5) / ModelDesc["num_io"]

    # print("len(test_arr_x5): %d" % len(test_arr_x5))
    # print("num_test_vectors: %d" % num_test_vectors)

    test_arr_x5 = np.array(test_arr_x5).reshape(1, -1, ModelDesc["num_io"])

    # print("len(test_arr_x5): %d" % len(test_arr_x5))
    # print("num_test_vectors: %d" % num_test_vectors)

    # print(test_arr_x5)

    # we feed it 4 inputs, so we can ask for anything a starting point anywhere between 0 -> -5
    # last possible X_batch: -5 -> -2
    # last possible y_batch: -4 -> -1

    smallest_rand_start = 0
    largest_rand_start = int(num_test_vectors - ModelDesc["num_timesteps"] - 1)

    random_start = np.random.randint(smallest_rand_start, largest_rand_start)
    # print("random_start: %d" % random_start)

    X_batch = test_arr_x5[:, random_start:random_start + ModelDesc["num_timesteps"], :]
    y_batch = test_arr_x5[:, random_start + 1:random_start + ModelDesc["num_timesteps"] + 1, :]

    return X_batch, y_batch






############## BEGIN MAKING TRAINING ##########


with tf.Session() as sess:
    sess.run(init)

    mse_list = np.zeros(int(MyModelDesc["num_iterations"] / 100) + 1)
    print(mse_list)

    for iteration in range(MyModelDesc["num_iterations"]):
        iter_div = int(iteration / 100)

        X_batch, y_batch = get_next_batch(MyModelDesc)
        print("X_batch:")
        print(X_batch)
        print("y_batch:")
        print(y_batch)

        # X_batch, y_batch = next_batch(training_set, batch_size, num_timesteps)
        sess.run(train, feed_dict={X_placeholder: X_batch, y_placeholder: y_batch})

        if iteration % 100 == 0:

            mse_list[iter_div] = loss.eval(feed_dict={X_placeholder: X_batch, y_placeholder: y_batch})

            if mse_list[iter_div - 1] < mse_list[iter_div]:
                print("%f < %f" % (mse_list[iter_div - 1], mse_list[iter_div]))
                learning_rate = MyModelDesc["learning_rate"] / 2
                print("Halving learning rate: %f to %f" % (MyModelDesc["learning_rate"] * 2, MyModelDesc["learning_rate"]))

            print(iteration, "\tMSE:", mse_list[iter_div])
            # print(mse_list)

    # Save Model for Later
    saver.save(sess, "../data/wave_predictor_rnn")

x_plt = np.arange(0, int(MyModelDesc["num_iterations"] / 100) + 1, 1).reshape(-1, 1)
y_plt = mse_list.reshape(-1, 1)
print(x_plt)
print(y_plt)

plt.show()
plt.plot(x_plt, y_plt, "r-")
plt.show()






############## BEGIN MAKING PREDICTIONS ##########


with tf.Session() as sess:
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "../data/wave_predictor_rnn")

    X_seed, y_true = get_next_batch(MyModelDesc)
    y_pred = sess.run(tf.nn.softmax(logits=outputs), feed_dict={X_placeholder: X_seed})
    print("\n")
    print(X_seed)

    print(y_true)
    print(y_pred)

    y_true_translated = []
    y_pred_translated = []

    for row in y_true[0]:
        y_true_translated.append(row[:].argmax())

    for row in y_pred[0]:
        y_pred_translated.append(row[:].argmax())

    print(y_true_translated)
    print(y_pred_translated)


    plt.plot(range(0,6), y_true_translated,"g-")
    plt.plot(range(0, 6), y_pred_translated, "ro")
    plt.show()

    print(np.array(X_seed[0, 0, :]).argmax())
    print(np.array(X_seed[0, 1, :]).argmax())
    print(np.array(X_seed[0, 2, :]).argmax())
    print(np.array(X_seed[0, 3, :]).argmax())
    print(np.array(X_seed[0, 4, :]).argmax())
    print(np.array(X_seed[0, 5, :]).argmax())

sys.exit()


"""