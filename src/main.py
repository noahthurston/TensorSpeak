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


# TRAINING
def train_it():
    vocab_size = 3000
    num_timesteps = 3
    num_layers = 3
    num_neurons_inlayer = 200
    learning_rate = 0.01
    #num_iterations = 1
    num_sentences_to_train = 10000
    model_name = ""
    graph_name = ""

    corpus_file_name = "trump_1k_tweets"

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
    model_name = "trump_1k_tweets_04-08--11-11"
    graph_name = "trump_1k_tweets_04-08--11-23"

    tmp = Model("_", num_io=12, num_timesteps=3, num_layers=5, num_neurons_inlayer=50,
                       learning_rate=0.001, batch_size=1)

    model = tmp.load(model_name)
    model.print_model_info()

    model.generate_sentences(graph_name, "trump")

def check_model(model_name):
    tmp = Model("rando", num_io=12, num_timesteps=3, num_layers=5, num_neurons_inlayer=50,
                       learning_rate=0.001, num_iterations=1, batch_size=1)

    model = tmp.load(model_name)
    model.print_model_info()


train_it()
#generate_it()
