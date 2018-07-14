"""
Loads a model from the models dir and generates a sentence.
Written by Noah Thurston
"""

import tensorflow as tf
from sentence_writer_rnn import Model

# define the model to be loaded, assumes it is in "../models/"
MODEL_NAME = "trump_2014-curr_cleaned_07-11--03-03"
GRAPH_NAME = "trump_2014-curr_cleaned_07-11--03-03"

# define temperature when sampling of the softmax distribution
TEMPERATURE = 0.01
STARTING_STRING = "the people"

# create a temporary model to load the saved model from
tmp = Model("_", num_io=12, num_timesteps=3, num_layers=5, num_neurons_inlayer=50,
                   learning_rate=0.001, batch_size=1)
model = tmp.load(MODEL_NAME)

# reset the tf graph in case an old one is in memory and generate a sentence
tf.reset_default_graph()
generated_sentence = model.generate_sentences(GRAPH_NAME, STARTING_STRING, TEMPERATURE)

print(generated_sentence)



