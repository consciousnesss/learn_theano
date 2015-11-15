#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy as np
from learn_theano.utils.display_filters import tile_raster_images
from learn_theano.utils.download_all_datasets import get_dataset
from theano.tensor.shared_randomstreams import RandomStreams
import pickle
import time
import cv2


def mean_cross_entropy(prediction_probailities, labels):
    return -T.mean(T.sum(labels*T.log(prediction_probailities) + (1-labels)*T.log(1-prediction_probailities), axis=1))


def one_zero_loss(prediction_labels, labels):
    return T.mean(T.neq(prediction_labels, labels))


def negative_log_likelihood_loss(prediction_probailities, labels):
    return -T.mean(T.log(prediction_probailities)[T.arange(labels.shape[0]), labels])


def load_dataset(dataset):
    set_x = theano.shared(np.asarray(dataset[0], dtype=theano.config.floatX), borrow=True)
    set_y = theano.shared(np.asarray(dataset[1], dtype=theano.config.floatX), borrow=True)
    return set_x, T.cast(set_y, 'int32')


def sigmoid_layer(input, n_in, n_out, rng):
    w_init = rng.uniform(
        low=-4.*np.sqrt(6./(n_in + n_out)),
        high=4.*np.sqrt(6./(n_in + n_out)),
        size=(n_in, n_out))
    W = theano.shared(
        np.asarray(w_init, dtype=theano.config.floatX),
        name='W',
        borrow=True)
    b = theano.shared(
        np.zeros((n_out,), dtype=theano.config.floatX),
        name='b',
        borrow=True
    )
    return T.nnet.sigmoid(T.dot(input, W)+b), [W, b]


def logistic_layer(input, n_in, n_out):
    W = theano.shared(
        np.zeros((n_in, n_out), dtype=theano.config.floatX),
        name='W',
        borrow=True)
    b = theano.shared(
        np.zeros((n_out,), dtype=theano.config.floatX),
        name='b',
        borrow=True
    )
    return T.nnet.softmax(T.dot(input, W)+b), [W, b]


def autoencoder(input, n_visible, W, b_hidden):
    b_visible = theano.shared(
        np.zeros((n_visible,), dtype=theano.config.floatX),
        name='b_visible',
        borrow=True
    )
    hidden_output = T.nnet.sigmoid(T.dot(input, W)+b_hidden)
    reconstructed = T.nnet.sigmoid(T.dot(hidden_output, W.T)+b_visible)
    return reconstructed, [W, b_hidden, b_visible]


def stacked_autoencoders(input, n_in, n_out, hidden_layers_sizes, rng):
    current_input = input
    current_input_size = n_in
    params = []
    for hidden_layer_size in hidden_layers_sizes:
        layer_output, (W, b_hidden) = sigmoid_layer(current_input, current_input_size, hidden_layer_size, rng)
        params += [W, b_hidden]

        # this autoencoder shares weights and biases with the sigmoid layer (reconstructing biases)
        reconstructed_out, params = autoencoder(current_input, current_input_size, W, b_hidden)

        current_input = layer_output
        current_input_size = hidden_layer_size

    output_layer, output_layer_params = logistic_layer(current_input, current_input_size, n_out)
    params += output_layer_params
    return output_layer, params


def autoencoder_costs(input, reconstructed_output, corruption_level, learning_rate, theano_rng):
    """ This function computes the cost and the updates for one trainng
    step of the dA """

    corrupted_input = theano_rng.binomial(size=input.shape, n=1,
                                          p=1-corruption_level, dtype=theano.config.floatX)*input
    cost = mean_cross_entropy(reconstructed_output, input)

    # compute the gradients of the cost of the `dA` with respect
    # to its parameters
    gparams = T.grad(cost, self.params)
    # generate the list of updates
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(self.params, gparams)
    ]

    return (cost, updates)


def run_4_stacked_autoencoder(corruption_level=0.3):
    mnist_pkl = get_dataset('mnist')
    with open(mnist_pkl) as f:
        train_set, valid_set, test_set = pickle.load(f)

    batch_size = 20
    learning_rate = 0.01
    training_epochs = 250
    n_in=28*28
    hidden_layers_sizes=[500, 500]
    n_out=10
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    train_set_x, train_set_y = load_dataset(train_set)
    valid_set_x, valid_set_y = load_dataset(valid_set)
    test_set_x, test_set_y = load_dataset(test_set)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size

    # load the whole test and validation set
    test_batch_size = valid_set_x.get_value(borrow=True).shape[0]
    n_validation_batches = valid_set_x.get_value(borrow=True).shape[0]/test_batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]/test_batch_size

    x = T.matrix('x')
    y = T.ivector('y')

    output_layer_output, params = stacked_autoencoders(x, n_in=n_in, n_out=n_out, hidden_layers_sizes=hidden_layers_sizes, rng=rng)

    y_predict = T.argmax(output_layer_output, axis=1)

    finetune_cost = negative_log_likelihood_loss(output_layer_output, y)


    minibatch_index = T.iscalar('minibatch_index')

    corruption_level_sym = T.scalar('corruption')
    learning_rate_sym = T.scalar('learning_rate')

    #corrupted_input = theano_rng.binomial(size=x.shape, n=1, p=1-corruption_level, dtype=theano.config.floatX)*x
    #cost = mean_cross_entropy(reconstructed, x)


    train_model = theano.function(
        inputs=[minibatch_index],
        outputs=[cost],
        updates=[[p, p - learning_rate*T.grad(cost, p)]
                 for p in params],
        givens={
            x: train_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
        },
        profile=False
    )

    start_time = time.time()

    print('Going to run the training with floatX=%s' % (theano.config.floatX))
    for epoch in range(training_epochs):
        costs = []
        epoch_start_time = time.time()
        for minibatch_index in range(n_train_batches):
            costs.append(train_model(minibatch_index))
        print("Mean costs at epoch %d is %f%% (ran for %.1fs)" % (epoch, np.mean(costs), time.time() - epoch_start_time))

    total_time = time.time()-start_time
    print('The training code run %.1fs, for %d epochs, for with %f epochs/sec' % (total_time, epoch, epoch/total_time))

    filters = tile_raster_images(X=params[0].get_value(borrow=True).T,
                                 img_shape=(28, 28), tile_shape=(23, 22),
                                 tile_spacing=(1, 1))
    filters = cv2.resize(filters, dsize=None, fx=1., fy=1.)
    cv2.imshow('filters', filters)
    cv2.waitKey(-1)


if __name__ == "__main__":
    run_4_stacked_autoencoder(corruption_level=0.3)
