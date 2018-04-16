# preprocessing function

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

"""
Linux commands for combing out weird trump tweeting habits

to get rid of repeating periods:
cat trump_500_tweets.csv | tr -s '.' >> tmp.csv

to get rid of repeating hythens:
cat trump_500_tweets.csv | tr -s '-' >> tmp.csv

to get rid of hyperlinks:
sed 's/http:\/\/.*/ /' < trump_500_tweets.csv >> tp

"""

class Preprocessor(object):
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

    def load_sentences(self, corpus_file_name, num_timesteps, vocab_size, max_sent_len=70):
        self.vocab_size = vocab_size
        self.corpus_file_name = corpus_file_name
        self.current_save_name = self.corpus_file_name + "_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
        ### tokenize sentences, call create_dictionary

        print("Reading CSV file: %s" % corpus_file_name)

        # create string of start and end tokens to buffer either side of each sentence
        sentence_start_string = ""
        sentence_end_string = " " + sentence_end_token
        # testing with only 1 start token, then buffering with zero arrays
        # for x in range(self.num_timesteps):
        for x in range(num_timesteps):
            sentence_start_string = sentence_start_string + sentence_start_token + " "
            #sentence_end_string = sentence_end_string + " " + sentence_end_token


        with open("../data/" + corpus_file_name + ".csv", 'rt') as f:
            reader = csv.reader(f, skipinitialspace=True)
            # reader.next()

            # read all csv lines and filter out empty lines
            csv_lines = [x for x in reader]
            csv_lines_filtered = filter(None, csv_lines)

            # tokenize sentences and attach start/end tokens
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in csv_lines_filtered])
            sentences = ["%s %s %s" % (sentence_start_string, sent, sentence_end_string) for sent in sentences]

        # print(sentences)

        # tokenize sentences into words using TweetTokenizer to preserve handles
        tk = nltk.TweetTokenizer(strip_handles=False, reduce_len=False, preserve_case=False)
        self.tokenized_sentences = [tk.tokenize(sent) for sent in sentences]

        # find max sentence length
        max_sent_rec = 0
        for i, sent in enumerate(self.tokenized_sentences):
            if len(self.tokenized_sentences[i]) > max_sent_rec:
                max_sent_rec = len(self.tokenized_sentences[i])
        print("Longest sentence is %d words" % (max_sent_rec))

        if max_sent_len > 0:
            # get rid of sentences longer than max_sent_len, optional argument
            total_num_sentences_untrimmed = len(self.tokenized_sentences)
            self.tokenized_sentences = [sent for sent in self.tokenized_sentences if len(sent) <= (max_sent_len)]
            print("%d out of %d sentences are %d-words-long or less." % (
                len(self.tokenized_sentences), total_num_sentences_untrimmed, max_sent_len))

        # create dictionary of words
        self.create_dictionary()

        # replace all words not in our vocabulary with the unknown token
        # add start and end sentence tokens
        for i, sent in enumerate(self.tokenized_sentences):
            self.tokenized_sentences[i] = [w if w in self.word_to_index else unknown_token for w in sent]

        # print(tokenized_sentences)
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

    # deprecated load dictionary function
    """
    def load_dictionary(self, dictionary_name):
        print("loading dictionary from: %s" % self.current_save_name)

        with open("../models/" + dictionary_name + "_INDEX_TO_WORD_" + '.pkl', 'rb') as f:
            self.index_to_word = pickle.load(f)
        with open("../models/" + dictionary_name + "_WORD_TO_INDEX" + '.pkl', 'rb') as f:
            self.word_to_index = pickle.load(f)
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


def test_save():
    preproc = Preprocessor()

    #def load_sentences(self, corpus_file_name, num_timesteps, vocab_size, max_sent_len=70):
    preproc.load_sentences("noahs_intro", 3, 12)

    #def index_sentences(self):
    indexed_sentences = preproc.index_sentences()

    print("Word to Index:")
    print(preproc.word_to_index)
    print("\n\n")

    print("Index to Word:")
    print(preproc.index_to_word)
    print("\n\n")

    print("Tokenized Sentences:")
    print(preproc.tokenized_sentences)
    print("\n\n")

    print("Indexed Sentences:")
    print(preproc.indexed_sentences)
    print("\n\n")

    preproc.save()

def test_load():
    print("loading test")

    tmp = Preprocessor()
    loaded_preproc = tmp.load("noahs_intro_03-26--14-19")

    print("Word to Index:")
    print(loaded_preproc.word_to_index)
    print("\n\n")

    print("Index to Word:")
    print(loaded_preproc.index_to_word)
    print("\n\n")

    print("Tokenized Sentences:")
    print(loaded_preproc.tokenized_sentences)
    print("\n\n")

    print("Indexed Sentences:")
    print(loaded_preproc.indexed_sentences)
    print("\n\n")
