"""
Defines the Preprocessor object for loading data and parsing it.
Written by Noah Thurston
"""

import numpy as np
import csv
import itertools
import nltk
#nltk.download('punkt')
import pickle
import datetime

unknown_token = "unknown_token"
sentence_start_token = "sentence_start"
sentence_end_token = "sentence_end"


class Preprocessor(object):
    """
    Preprocessor class has attributes that define the vocab size and the file being parsed.
    Has methods to clean the data, tokenize words into a dictionary and save the parsed data.
    """
    def __init__(self):
        self.corpus_file_name = ""
        self.current_save_name = ""
        self.vocab_size = -1

        self.current_save_name = self.corpus_file_name + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")

        # variables initialized later
        self.tokenized_sentences = []
        self.indexed_sentences = []
        self.word_to_index = []
        self.index_to_word = []

    def load_sentences(self, corpus_file_name, num_timesteps, vocab_size, max_sent_len=30, min_sent_len=8):
        self.vocab_size = vocab_size
        self.corpus_file_name = corpus_file_name
        self.current_save_name = self.corpus_file_name + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
        ### tokenize sentences, call create_dictionary

        print("Reading CSV file: %s" % corpus_file_name)

        # create string of start and end tokens to buffer either side of each sentence
        sentence_start_string = ""
        sentence_end_string = ""
        # testing with only 1 start token, then buffering with zero arrays
        # for x in range(self.num_timesteps):
        for x in range(num_timesteps):
            sentence_start_string = sentence_start_string + sentence_start_token + " "
            sentence_end_string = sentence_end_string + " " + sentence_end_token


        with open("../data/" + corpus_file_name + ".csv", 'rt') as f:
            #!reader = csv.reader(f, skipinitialspace=True)
            # reader.next()

            # read all csv lines and filter out empty lines
            #!csv_lines = [x for x in reader]
            #!csv_lines_filtered = filter(None, csv_lines)

            reader = csv.reader(f, skipinitialspace=True)
            #reader.next()

            # tokenize sentences and attach start/end tokens
            #sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in csv_lines_filtered])
            #sentences = ["%s %s %s" % (sentence_start_string, sent, sentence_end_string) for sent in sentences]

            # tokenize sentences and attach start/end tokens
            #sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in csv_lines_filtered])
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
            sentences = ["%s %s %s" % (sentence_start_string, sent, sentence_end_string) for sent in sentences]

        # print(sentences)

        # tokenize sentences into words using TweetTokenizer to preserve handles
        #! tk = nltk.TweetTokenizer(strip_handles=False, reduce_len=False, preserve_case=False)
        #! self.tokenized_sentences = [tk.tokenize(sent) for sent in sentences]
        self.tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # find max sentence length
        max_sent_rec = 0
        for i, sent in enumerate(self.tokenized_sentences):
            if len(self.tokenized_sentences[i]) > max_sent_rec:
                max_sent_rec = len(self.tokenized_sentences[i])
        print("Longest sentence is %d words" % (max_sent_rec))

        if max_sent_len > 0:
            # get rid of sentences longer than max_sent_len, optional argument
            total_num_sentences_untrimmed = len(self.tokenized_sentences)
            self.tokenized_sentences = [sent for sent in self.tokenized_sentences if (min_sent_len <= len(sent) <= max_sent_len)]
            print("%d out of %d sentences are between %d and %d words long in length." % (
                len(self.tokenized_sentences), total_num_sentences_untrimmed, min_sent_len, max_sent_len))

        # create dictionary of words
        self.create_dictionary()

        # replace all words not in our vocabulary with the unknown token
        # add start and end sentence tokens
        for i, sent in enumerate(self.tokenized_sentences):
            self.tokenized_sentences[i] = [w if w in self.word_to_index else unknown_token for w in sent]

        #print(self.tokenized_sentences)
        #return self.tokenized_sentences

    def create_dictionary(self):
        print("creating dictionary")

        # create dictionary of words
        word_freq = nltk.FreqDist(itertools.chain(*self.tokenized_sentences))
        print("Found %d unique words." % len(word_freq.items()))
        vocab = word_freq.most_common(self.vocab_size - 1)
        self.index_to_word = [x[0] for x in vocab]
        self.index_to_word.append(unknown_token)
        self.word_to_index = dict([w, i] for i, w in enumerate(self.index_to_word))

        print("Using a vocab of %d words." % self.vocab_size)

        """
        # save index_to_word and word_to_index as pkls
        with open("../models/" + self.current_save_name + "_INDEX_TO_WORD_" + '.pkl', 'wb') as f:
            pickle.dump(self.index_to_word, f, pickle.HIGHEST_PROTOCOL)
        with open("../models/" + self.current_save_name + "_WORD_TO_INDEX" + '.pkl', 'wb') as f:
            pickle.dump(self.word_to_index, f, pickle.HIGHEST_PROTOCOL)
        """

    def index_sentences(self):
        #vectorized_sentences = []

        for sentence in self.tokenized_sentences:
            indexed_sentence = []
            #vectorized_sentence = np.array([])
            for word_str in sentence:
                word_index = self.word_to_index[word_str]
                #vectorized_word = np.zeros(self.vocab_size).tolist()
                #vectorized_word[word_index] = 1

                indexed_sentence = np.append(indexed_sentence, word_index)

            self.indexed_sentences.append(indexed_sentence)

        return self.indexed_sentences

    def update_save_name(self):
        self.current_save_name = self.corpus_file_name + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")

    def save(self):
        print("Saving preprocessor")
        # save preprocessor as a .pkl

        self.update_save_name()
        with open("../preprocessors/" + self.current_save_name + '.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        print("Loading preprocessor: %s" % filename)
        # load preprocessor as a .pkl

        with open("../preprocessors/" + filename + ".pkl", 'rb') as f:
            loaded_preprocessor = pickle.load(f)
        return loaded_preprocessor

    def filter_out_retweets(self):
        # this function reads in data downloaded from trumptwitterarchive.com
        # and does some basic cleaning
        file_dir = "../data/"
        filename = "trump_2014-curr"

        read_file = open(file_dir + filename + ".csv")
        write_file = open(file_dir + filename + "_no_rts.csv", "w")

        # iterate through files, ignoring retweets and quote-tweets
        for line in read_file:
            if (line[0:2] != 'RT') and (line[0] != '“') and (line[0] != '"'):
                write_file.write(line)

        read_file.close()
        write_file.close()

        """
        Data is cleaned further with these bash commands:

        to get rid of repeating periods:
        cat trump_2014-curr_no_rts.csv | tr -s '.' >| tmp1.csv

        to get rid of repeating hyphens:
        cat tmp1.csv | tr -s '-' >| tmp2.csv

        to get rid of hyperlinks:
        sed 's/http:\/\/.*/ /' < tmp2.csv >| tmp3.csv
        sed 's/https:\/\/.*/ /' < tmp3.csv >| trump_2014-curr_cleaned.csv
        """

