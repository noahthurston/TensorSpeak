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

ted talk with 100 sentences:
2500 at 0.01 (ted_talk_100_04-16--19-38), finished around 4, repeating "down"
5000 at 0.01 (ted_talk_100_04-16--19-48), finished around 2.5, repeating "left-handers"
5000 at 0.005 (ted_talk_100_04-16--20-08), finished around 1.5, still repeatig "left-handers"
5000 at 0.0025 (ted_talk_100_04-16--20-24), finished around 1.5, still repeatig "left-handers"


ted talk with 100 sentences AND n-timestep end sentence tokens
5000 at 0.01 (ted_talk_100_04-16--21-09), finished around 3, ends immediately
2500 at 0.01, (ted_talk_100_04-16--21-16), finished around 2.25, repeating true
3000 at 0.005, (ted_talk_100_04-16--21-23), finished around 1.75, ends immediately
2500 at 0.001, (ted_talk_100_04-16--21-30), finished around 1.25, sometimes ends, sometimes never ends
2500 at 0.00025, (ted_talk_100_04-16--21-36), finsihed around 1.25, sometimes ends, sometimes never ends


plan for over night:
-use n timesteps as end tokens
-train on 1k with almost full vocab utilization: 3000/3304
-train for 100k sentences:  ~5 hours
-halve LR every 20k sentences


1000 / 165 seconds
6 / 1 second
130,000 / 21,600 seconds
6 hrs = 21,600 seconds


ID: 0001
Not changing the learning rate and training for 100k sentences worked!
trump_1k_tweets_04-17--14-48:
    -1k tweets, 3000/3304 vocab, sentences_len < 70 words
    -LR=0.001 for 100k
    -5 timesteps
    -3 layers, 100 nodes each
    -ended around 2
    -relatively good, usually end sentences, sometimes doesnt
    -mostly incoherent word combinations

Now training with 3 timesteps:
trump_1k_tweets_04-17--22-15:
    -1k tweets, 3000/3304 vocab, sentences_len < 70 words
    -LR=0.001 for 100k
    -3 timesteps
    -3 layers, 100 nodes each
    -ended around 2.8
    -not ending sentences, 3 TS might be too short
    
    
Training on 10k tweets with 5 timesteps (took about 30 hours)
trump_10k_tweets_04-19--05-50:
    -10k tweets, 13.3k sentences, 10k/13.4k vocab, sentences_len < 30 words
    -LR=0.001 for 200k
    -5 timesteps
    -3 layers, 200 nodes each
    -ended around 5
    -just keeps repeating sequences of "the" "republican" and "." 


Now with only 100 nodes / layer
Training on 10k tweets with 5 timesteps (took about 30 hours)
trump_10k_tweets_04-19--15-55:
    -10k tweets, 13.3k sentences, 10k/13.4k vocab, sentences_len < 30 words
    -LR=0.001 for 106k, stopped early because it was plateauing
    -5 timesteps
    -3 layers, 100 nodes each
    -ended around 5
    -similar behavior as before, repeating " . will be the the . " 

Trying again, but with LR of 0.01 instead of 0.001
Maybe it was stuck in local minimum
trump_10k_tweets_04-19--20-27
    -trained for 40k senteces, but was diverging hard so I stopped it early
    -just repeating "should"



Going back to original architecture, but with 2k tweets
trump_2k_tweets_04-20--10-10:
    -1k tweets, 4000/4365 vocab, 2.5k sentences < 30 words,
    -LR=0.001 for 200k
    -5 timesteps
    -3 layers, 100 nodes each
    -ended around 3
    -mostly full sentences, but its stringing them together and using random punctuation
    -check preprocessing for errors?

Need to move to AWS for training

Trying now with 200 neurons per layer


95k: trump_2k_tweets_04-20--23-54
100k: trump_2k_tweets_04-21--00-21
150k: trump_2k_tweets_04-21--04-45
200k: trump_2k_tweets_04-21--09-11 (-06)


Re-doing #0001, but with sentence loss, seeing if the memory can survive
trump_1k_tweets_04-17--14-48:
    -1k tweets, 3000/3304 vocab, sentences_len < 70 words
    -LR=0.001 for 100k
    -5 timesteps
    -3 layers, 100 nodes each
    -ended around 2
    -relatively good, usually end sentences, sometimes doesnt
    -mostly incoherent word combinations



NEXT:
    vocab_size = 4000
    num_timesteps = 4
    num_layers = 3
    num_neurons_inlayer = 150
    learning_rate = 0.001
    num_sentences_to_train = 200*1000
    model_name = ""
    graph_name = ""



Calculating sentence error problems:
    -thought it worked efficiently, but then only did 20k sentences / 8 hours, should've done around 80k
    -cpu cores were oscillating and hitting 100% usage
    -only ~16% free ram
    -graphics card was mostly at 6%, sometimes flicker to 50%

322k words / 21k sentences
~15.3 words / sentence
20k sentences / 8 hours


    vocab_size = 4000
    num_timesteps = 4
    num_layers = 3
    num_neurons_inlayer = 150
    learning_rate = 0.001
    num_sentences_to_train = 200*1000
    model_name = ""
    graph_name = ""

    corpus_file_name = "trump_2k_tweets"


"""

# TRAINING
def train_it():
    vocab_size = 500
    num_timesteps = 4
    num_layers = 3
    num_neurons_inlayer = 100
    learning_rate = 0.001
    num_sentences_to_train = 200*1000
    model_name = ""
    graph_name = ""

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
def generate_it(model_name="trump_2k_tweets_04-22--21-44", graph_name="trump_2k_tweets_04-22--21-44"):
    #model_name = "trump_2k_tweets_04-20--10-07"
    #graph_name = "trump_2k_tweets_04-20--10-10"

    tmp = Model("_", num_io=12, num_timesteps=3, num_layers=5, num_neurons_inlayer=50,
                       learning_rate=0.001, batch_size=1)

    model = tmp.load(model_name)
    # model.print_model_info()

    model.generate_sentences(graph_name, "the")

def check_model(model_name):
    tmp = Model("rando", num_io=12, num_timesteps=3, num_layers=5, num_neurons_inlayer=50,
                       learning_rate=0.001, num_iterations=1, batch_size=1)

    model = tmp.load(model_name)
    model.print_model_info()


# train_it()

for _ in range(1):
    generate_it()


#generate_it(model_name="trump_2k_tweets_04-21--08-25", graph_name="trump_2k_tweets_04-21--08-25")
#generate_it(model_name="trump_2k_tweets_04-21--09-06", graph_name="trump_2k_tweets_04-21--09-11")



"""
end of last one: 
sentence_start where is the endorsement unemployment is a workers than a ( products off . ? '' ( books . debt debt . '' sentence_end

MIA:
95k: trump_2k_tweets_04-20--23-54
100k: trump_2k_tweets_04-21--00-21
150k: trump_2k_tweets_04-21--04-45


from an hour before finishing:
sentence_start where is sentence_start sentence_start sentence_start sentence_start sentence_start sentence_start republicans sentence_start sentence_start


200k: trump_2k_tweets_04-21--09-11 (-06)
sentence_start where is @ is @ unknown_token unknown_token . with with on @ @ at on on with with on on on on on at on on on on @ on on on on on on on on on on on


dip around 1.55M at 100.3K sentence:
trump_2k_tweets_04-22--16-50

lowest sentence loss at 120k: 
trump_2k_tweets_04-22--17-58
100k took 7 hours
-




"""