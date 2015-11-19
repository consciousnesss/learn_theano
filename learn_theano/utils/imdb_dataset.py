from learn_theano.utils.s3_download import S3
import cPickle

import numpy as np
import os


def get_original_imdb_pkl():
    return S3().download('datasets/imdb.pkl.gz')


def get_imdb_words_to_numbers_dict():
    filename = S3().download('datasets/imdb.dict.pkl.gz')
    with open(filename, 'rb') as f:
        return cPickle.load(f)


def get_imdb_number_to_words_dict():
    forward = get_imdb_words_to_numbers_dict()
    return {v: k for k, v in forward.iteritems()}


def load_full_imdb(vocabulary_size=10000, validation_portion=0.05, maximum_sequence_length=100,
                   test_set_size=500):
    '''
    imdb dataset processed by this function with default parameters is uploaded to S3: imdb_filtered.pkl.gz
    '''
    filename = get_original_imdb_pkl()
    print(filename)
    with open(filename, 'rb') as f:
        train_set = cPickle.load(f)
        test_set = cPickle.load(f)

    train_inputs, train_labels = train_set
    test_inputs, test_labels = test_set

    valid_sequences = [i for i, x in enumerate(train_inputs) if len(x) <= maximum_sequence_length]
    train_inputs, train_labels = np.array(train_inputs)[valid_sequences], np.array(train_labels)[valid_sequences]

    valid_sequences = [i for i, x in enumerate(test_inputs) if len(x) <= maximum_sequence_length]
    test_inputs, test_labels = np.array(test_inputs)[valid_sequences], np.array(test_labels)[valid_sequences]

    def filter_words(x):
        return [[1 if w >= vocabulary_size else w for w in sequence] for sequence in x]

    train_inputs = filter_words(train_inputs)
    test_inputs = filter_words(test_inputs)

    n_validation = int(len(train_inputs)*validation_portion)
    valid_inputs, valid_labels = train_inputs[-n_validation:], train_labels[-n_validation:]
    train_inputs, train_labels = train_inputs[:-n_validation], train_labels[:-n_validation]

    test_inputs, test_labels = test_inputs[:test_set_size], test_labels[:test_set_size]

    def sort_by_length(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    sorted_index = sort_by_length(train_inputs)
    train_inputs = [train_inputs[i] for i in sorted_index]
    train_labels = [train_labels[i] for i in sorted_index]

    sorted_index = sort_by_length(valid_inputs)
    valid_inputs = [valid_inputs[i] for i in sorted_index]
    valid_labels = [valid_labels[i] for i in sorted_index]

    sorted_index = sort_by_length(test_inputs)
    test_inputs = [test_inputs[i] for i in sorted_index]
    test_labels = [test_labels[i] for i in sorted_index]

    train_set = (train_inputs, train_labels)
    valid_set = (valid_inputs, valid_labels)
    test_set = (test_inputs, test_labels)

    return train_set, valid_set, test_set


if __name__ == "__main__":
    train_set, valid_set, test_set = load_full_imdb()
    print(len(train_set[0]), len(valid_set[0]), len(test_set[0]))
    with open(os.path.expanduser('~/imdb_filtered.pkl'), 'w') as f:
        cPickle.dump((train_set, valid_set, test_set), f, cPickle.HIGHEST_PROTOCOL)
