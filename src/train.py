"""
Creates model or loads it from models dir and
trains it after preprocessing a corpus of text
Written by Noah Thurston
"""

from sentence_writer_rnn import Model
from preprocessing import Preprocessor

# model hyper parameters
vocab_size = 8000
num_timesteps = 3
num_layers = 3
num_neurons_inlayer = 200
learning_rate = 0.0001
num_sentences_to_train = 450*1000

# for loading a model to continue training,
model_name = ""
graph_name = ""

# assumes corpus is in ../data/
corpus_file_name = "trump_2014-curr_cleaned"

model = Model(corpus_file_name=corpus_file_name, num_io=vocab_size, num_timesteps=num_timesteps, num_layers=num_layers, num_neurons_inlayer=num_neurons_inlayer,
                   learning_rate=learning_rate, batch_size=1)

# start training new model
if graph_name == "":
    preprocessor = Preprocessor()
    preprocessor.load_sentences(corpus_file_name, num_timesteps, vocab_size)
    model.indexed_sentences = preprocessor.index_sentences()
    model.word_to_index, model.index_to_word = preprocessor.word_to_index, preprocessor.index_to_word
    preprocessor.save()
    model.print_model_info()
    model.train_model(num_sentences_to_train)

# load old model and continue training (not ideal for Adam optimization)
else:
    model = model.load(model_name)
    model.print_model_info()
    model.learning_rate = learning_rate

    model.train_model(num_sentences_to_train, save_every=10000, graph_name=graph_name)
