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

from sentence_writer_rnn import Model
from preprocessing import Preprocessor


"""
Training history:


training on 100, vocab of 700/703, 3 timesteps, 3 layers, 200 neurons in layer, 1 end token:
2600 at 0.01 (trump_100_tweets_04-15--20-46) finished around 4.5, kept repeating "the"
2500 at 0.001 (trump_100_tweets_04-15--20-58) finished around 3.5, still repeating "the"
2500 at 0.001 (trump_100_tweets_04-15--21-12) finished around 3.0, still repeating "the"
2500 at 0.0001 (trump_100_tweets_04-15--21-24) finished around 2.6, still repeating "the" 
2500 at 0.00001 (trump_100_tweets_04-15--21-33) finished around 2.6, still repeating "the"
After bringing it all the way down to 2.6, still repeating "the" and stuck at 2.6

training on 100, vocab of 700/703, 3 timesteps, 3 layers, 100 neurons in layer, 1 end token:
2500 at 0.01 (trump_100_tweets_04-15--21-43), finished around 5, repeating "the"
2500 at 0.01 (trump_100_tweets_04-15--21-53), finished around 4, repeating "the"
2500 at 0.01 (trump_100_tweets_04-15--22-01), finished around 3.75, repeating "be"
2500 at 0.001 (trump_100_tweets_04-15--22-09), finished around 3.1, repeating "was"
2500 at 0.0001 (trump_100_tweets_04-15--22-18), finished around 3.0, repeating "was"

Neither of these designs were successful, need to try:
-adding back n-timesteps of end sentence tokens
-increasing to 5 layers or more nodes/layer (cant handle the complexity)
-increasing timesteps back to 5


"""

# TRAINING
def train_it():
    vocab_size = 700
    num_timesteps = 3
    num_layers = 3
    num_neurons_inlayer = 100
    learning_rate = 0.0001
    #num_iterations = 1
    num_sentences_to_train = 2500
    model_name = "trump_100_tweets_04-15--22-09"
    graph_name = "trump_100_tweets_04-15--22-09"

    corpus_file_name = "trump_100_tweets"

    model = Model(corpus_file_name=corpus_file_name, num_io=vocab_size, num_timesteps=num_timesteps, num_layers=num_layers, num_neurons_inlayer=num_neurons_inlayer,
                       learning_rate=learning_rate, batch_size=1)

    if graph_name == "":
        preprocessor = Preprocessor()
        preprocessor.load_sentences(corpus_file_name, num_timesteps, vocab_size)
        model.indexed_sentences = preprocessor.index_sentences()
        model.word_to_index, model.index_to_word = preprocessor.word_to_index, preprocessor.index_to_word
        preprocessor.save()
        model.print_model_info()
        model.train_model(num_sentences_to_train)

    else:
        model = model.load(model_name)
        model.print_model_info()
        model.learning_rate = learning_rate

        model.train_model(num_sentences_to_train, save_every=300, graph_name=graph_name)

# GENERATING
def generate_it():
    model_name = "trump_100_tweets_04-15--22-18"
    graph_name = "trump_100_tweets_04-15--22-18"

    tmp = Model("_", num_io=12, num_timesteps=3, num_layers=5, num_neurons_inlayer=50,
                       learning_rate=0.001, batch_size=1)

    model = tmp.load(model_name)
    model.print_model_info()

    model.generate_sentences(graph_name, "where")

def check_model(model_name):
    tmp = Model("rando", num_io=12, num_timesteps=3, num_layers=5, num_neurons_inlayer=50,
                       learning_rate=0.001, num_iterations=1, batch_size=1)

    model = tmp.load(model_name)
    model.print_model_info()



#train_it()
generate_it()



