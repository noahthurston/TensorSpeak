import numpy as np
import csv
import itertools
import nltk


def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print
    "Saved model parameters to %s." % outfile


def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print
    "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])


def load_data(file_name, vocab_size=8000, max_sent_len=32):
    unknown_token = "unknown_token"
    sentence_start_token = "sentence_start"
    sentence_end_token = "sentence_end"

    print("Reading CSV file: %s" % file_name)

    with open(file_name, 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()

        # read all csv lines and filter out empty lines
        csv_lines = [x for x in reader]
        csv_lines_filtered = filter(None, csv_lines)

        # tokenize sentences and attach start/end tokens
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in csv_lines_filtered])
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

    # tokenize sentences into words using TweetTokenizer to preserve handles
    tk = nltk.TweetTokenizer(strip_handles=False, reduce_len=False, preserve_case=False)
    tokenized_sentences = [tk.tokenize(sent) for sent in sentences]

    # find max sentence length
    max_sent_rec = 0
    for i, sent in enumerate(tokenized_sentences):
        # print (len(tokenized_sentences[i]))
        if len(tokenized_sentences[i]) > max_sent_rec:
            max_sent_rec = len(tokenized_sentences[i])
    print("Longest sentence is %d words" % (max_sent_rec))

    # get rid of sentences longer than max_sent_len
    total_num_sentences_untrimmed = len(tokenized_sentences)
    tokenized_sentences = [sent for sent in tokenized_sentences if len(sent) <= (max_sent_len)]
    print("%d out of %d sentences are %d-words-long or less." % (len(tokenized_sentences), total_num_sentences_untrimmed, max_sent_len))

    # create dictionary of words
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words." % len(word_freq.items()))
    vocab = word_freq.most_common(vocab_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([w, i] for i, w in enumerate(index_to_word))

    print("Using a vocab of %d words." % vocab_size)

    # replace words that are not within our vocab with the unknown_token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocab_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocab_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # Create the training data
    x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    return x_train, y_train, index_to_word, word_to_index

#### tests if this works on mac
# def load_data(file_name, vocab_size=8000, max_sent_len=32):
x_train, y_train, index_to_word, word_to_index = load_data("../data/test_sentences.csv", vocab_size = 5)

