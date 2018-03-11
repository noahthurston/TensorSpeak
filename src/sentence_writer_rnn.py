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

        self.historical_mse = np.array([0])

        self.corpuse_file_name = ""

        # variables initialized later
        # self.vectorized_sentences = []
        self.word_to_index = []
        self.index_to_word = []

    def train_model(self, vectorized_sentences):
        print("training started at: " + time.asctime(time.localtime(time.time())))

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

        print(self.word_to_index)


        ### TRAINING MODEL
        with tf.Session() as sess:
            sess.run(init)

            for iteration in range(self.num_iterations):
                print("BEGINNING ITERATION #" + str(iteration))
                mse_count = 0
                avg_mse = 0

                np.random.shuffle(vectorized_sentences)


                for sent_index, sentence in enumerate(vectorized_sentences):
                    #print("Sentence string: " + str(sentence))
                    # iter_div = int(iteration / 100)
                    #print("New sentence")
                    #sentence = [sentence_start_token for x in range(self.num_timesteps)] + sentence + [sentence_end_token for x in range(self.num_timesteps)]



                    for word in range(len(sentence)-self.num_timesteps):
                        X_batch = [sentence[word:word+self.num_timesteps]]
                        y_batch = [sentence[word+1:word+1+self.num_timesteps]]
                        #print("X_batch:\n" + str(X_batch))
                        #print("y_batch:\n" + str(y_batch) + '\n')


                        sess.run(train, feed_dict={X_placeholder: X_batch, y_placeholder: y_batch})

                        if sent_index % 100 == 0:
                            #print("X_batch:\n" + str(X_batch))
                            #print("y_batch:\n" + str(y_batch) + '\n')

                            curr_mse = loss.eval(feed_dict={X_placeholder: X_batch, y_placeholder: y_batch})
                            #print(curr_mse)
                            avg_mse = avg_mse + curr_mse
                            mse_count = mse_count + 1


                    if sent_index % 100 == 0:
                        avg_mse = avg_mse/mse_count
                        print("Sentence: " + str(sent_index))
                        print("Avg MSE: " + str(avg_mse))
                        self.historical_mse = np.append(self.historical_mse, avg_mse)
                        mse_count = 0
                        avg_mse = 0

            saver.save(sess, self.save_dir + "saved_model")

        self.graph_mse()

    def graph_mse(self):
        x_values = np.array(range(len(self.historical_mse)))

        title_str = ("%s, TS=%d, NeurPL=%d, LR=%.4f, Iters=%d" % (self.corpuse_file_name, self.num_timesteps, self.num_neurons_inlayer, self.learning_rate, self.num_iterations))
        plt.plot(x_values, self.historical_mse)
        plt.title(title_str)

        t = time.asctime(time.localtime(time.time()))
        save_str = self.save_dir + "graph_" + self.corpuse_file_name + "_" + t + "_.png"

        plt.savefig(save_str, format='png', dpi=300)



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


        print(self.word_to_index)


        ### GENERATING SENTENCES WITH MODEL
        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, self.save_dir + "saved_model")



            # vectorized version of a single sentence_start_token
            sentence_start_token_vectorized = np.zeros((self.vocab_size))
            #sentence_start_token_vectorized[self.word_to_index[sentence_start_token]] = 1

            sentence_start_token_vectorized[random.randint(0,self.vocab_size/10)] = 1


            for generated_sent_index in range(num_sentences):
                max_sentence_length = 20
                curr_sentence_length = 0

                #print("Feeding: %s" % self.index_to_word[curr_word_vector.argmax()])

                generated_sentence = np.zeros((1,1,self.vocab_size*2))
                generated_sentence[0, 0, self.word_to_index[sentence_start_token]] = 1
                generated_sentence[0, 0, self.vocab_size + random.randint(0,self.vocab_size/10)] = 1

                #print(generated_sentence.shape)

                generated_sentence = generated_sentence.reshape((1, -1, self.vocab_size))
                #print(generated_sentence.shape)

                curr_words_vector = generated_sentence[:, -self.num_timesteps:, :].reshape(1, self.num_timesteps, self.vocab_size)
                #print("curr_words_vector.shape " + str(curr_words_vector.shape))

                pred_words_vector = generated_sentence[:, -self.num_timesteps:, :]
                #print("pred_words_vector.shape " + str(pred_words_vector.shape))

                while (sentence_end_token != self.index_to_word[generated_sentence[0,-1,:].argmax()]) & (curr_sentence_length < max_sentence_length):
                    # feed in current word, predict next word
                    pred_words_vector = sess.run(tf.nn.softmax(logits=outputs), feed_dict={X_placeholder: curr_words_vector})
                    #print("pred_words_vector.shape " + str(pred_words_vector.shape))

                    generated_sentence = np.append(generated_sentence, pred_words_vector[:, -1, :]).reshape((1, -1, self.vocab_size))
                    #print("generated_sentence.shape " + str(generated_sentence.shape))

                    curr_words_vector = generated_sentence[:, -self.num_timesteps:, :]
                    #print("curr_words_vector.shape " + str(curr_words_vector.shape))


                    pred_word_str = self.index_to_word[generated_sentence[0, -1].argmax()]
                    fed_word_str = self.index_to_word[generated_sentence[0, -2].argmax()]
                    #print("Word #" + str(curr_sentence_length))
                    #print("Fed: " + fed_word_str)
                    #print("Predicted: " + pred_word_str)

                    #generated_sentence.append(self.index_to_word[pred_words_vector.argmax()])

                    curr_sentence_length = curr_sentence_length + 1


                translated_sentence = ""
                for word in generated_sentence[0,:,:]:
                    translated_sentence = translated_sentence + " " + self.index_to_word[word.argmax()]


                print('\t' + '\t' + '\t' + '\t' + '\t' + translated_sentence)



        #### target index = myList.index(max(myList))



    #def get_next_batch(self, vectorized_sentences, iteration):


    def load_sentences(self, corpus_file_name, max_sent_len = -1):

        self.corpuse_file_name = corpus_file_name
        ### tokenize sentences, call create_dictionary

        print("Reading CSV file: %s" % corpus_file_name)

        # create string of start and end tokens to buffer either side of each sentence
        sentence_start_string = ""
        sentence_end_string = ""
        #testing with only 1 start token, then buffering with zero arrays
        #for x in range(self.num_timesteps):
        for x in range(1):
            sentence_start_string = sentence_start_string + sentence_start_token + " "
            sentence_end_string = sentence_end_string + " " + sentence_end_token

        with open(self.save_dir + corpus_file_name + ".csv", 'rt') as f:
            reader = csv.reader(f, skipinitialspace=True)
            #reader.next()

            # read all csv lines and filter out empty lines
            csv_lines = [x for x in reader]
            csv_lines_filtered = filter(None, csv_lines)

            # tokenize sentences and attach start/end tokens

            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in csv_lines_filtered])
            sentences = ["%s %s %s %s" % (sentence_start_string, sent, sentence_end_string, sentence_end_string) for sent in sentences]

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
            #tokenized_sentences[i] = [sentence_start_token for x in range(self.num_timesteps)] + tokenized_sentences[i] + [sentence_end_token for x in range(self.num_timesteps)]



            #tokenized_sentences[i] = [np.zeros(self.vocab_size) for x in range(self.num_timesteps-1)] + sent + [np.zeros(self.vocab_size) for x in range(self.num_timesteps-1)]

        print(tokenized_sentences)
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
            #vectorized_sentence = [np.zeros(self.vocab_size).tolist() for x in range(self.num_timesteps-1)]
            vectorized_sentence = []
            for word_str in sentence:
                #innefficient way
                word_index = self.word_to_index[word_str]
                #vectorized_word = [1 if x == self.word_to_index[sentence_start_token] else 0 for x in range(self.vocab_size)]
                vectorized_word = np.zeros(self.vocab_size).tolist()
                vectorized_word[word_index] = 1

                vectorized_sentence.append(vectorized_word)

            #for x in range(self.num_timesteps-1):
            #    vectorized_sentence.append(np.zeros(self.vocab_size).tolist())

            #print("vectorized_sentence: " + str(vectorized_sentence))
            vectorized_sentences.append(vectorized_sentence)


        #print("START\n\n\n")
        #print(self.index_to_word[0])
        #print(self.index_to_word[1])
        #print(self.word_to_index[sentence_end_token])

        #print("vectorized_sentences[0]: " + str(vectorized_sentences[0]))
        return vectorized_sentences

# def __init__(self, num_io, num_timesteps, num_neurons_inlayer, learning_rate, num_iterations, batch_size, save_dir):



test_model = Model(num_io=3600, num_timesteps=1, num_neurons_inlayer=50,
                   learning_rate=0.001, num_iterations=50, batch_size=1, save_dir="../data/trump_model/")

#def load_sentences(self, corpus_path, max_sent_len = -1):


# TRAINING
def train():
    tokenized_sentences = test_model.load_sentences("trump_1k_tweets")
    #print(tokenized_sentences)
    vectorized_sentences = test_model.sentences_to_vectors(tokenized_sentences)
    #print(vectorized_sentences)
    test_model.train_model(vectorized_sentences)




# GENERATING
def generate():
    test_model.load_dictionary("trump_1k_tweets")
    test_model.generate_sentences(5)


#train()
generate()


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