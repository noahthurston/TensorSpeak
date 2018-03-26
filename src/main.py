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

    vocab_size = 12
    num_timesteps = 3
    num_layers = 5
    num_neurons_inlayer = 50
    learning_rate = 0.001
    num_iterations = 3

    corpus_file_name = "test_sentences"

    model = Model(corpus_file_name=corpus_file_name, num_io=vocab_size, num_timesteps=num_timesteps, num_layers=num_layers, num_neurons_inlayer=num_neurons_inlayer,
                       learning_rate=learning_rate, num_iterations=num_iterations, batch_size=1)

    preprocessor = Preprocessor()

    # def load_sentences(self, corpus_file_name, num_timesteps, vocab_size, max_sent_len=70):
    preprocessor.load_sentences(corpus_file_name, num_timesteps, vocab_size)
    model.indexed_sentences = preprocessor.index_sentences()

    model.word_to_index, model.index_to_word = preprocessor.word_to_index, preprocessor.index_to_word

    model.save()
    preprocessor.save()

    model.print_model_info()

    model.train_model()

# GENERATING
def generate_it():
    model_name = "test_sentences_03-26--15-36"
    graph_name = "test_sentences_03-26--15-37"

    tmp = Model("_", num_io=12, num_timesteps=3, num_layers=5, num_neurons_inlayer=50,
                       learning_rate=0.001, num_iterations=1, batch_size=1)

    model = tmp.load(model_name)
    model.print_model_info()

    model.generate_sentences(graph_name, "the quick")

def check_model(model_name):
    tmp = Model("rando", num_io=12, num_timesteps=3, num_layers=5, num_neurons_inlayer=50,
                       learning_rate=0.001, num_iterations=1, batch_size=1)

    model = tmp.load(model_name)
    model.print_model_info()


#train_it()
generate_it()