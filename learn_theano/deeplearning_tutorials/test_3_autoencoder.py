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


def load_dataset(dataset):
    set_x = theano.shared(np.asarray(dataset[0], dtype=theano.config.floatX), borrow=True)
    set_y = theano.shared(np.asarray(dataset[1], dtype=theano.config.floatX), borrow=True)
    return set_x, T.cast(set_y, 'int32')


def denoising_autoencoder(input, n_visible, n_hidden, rng):
    w_init = rng.uniform(
        low=-4.*np.sqrt(6./(n_visible + n_hidden)),
        high=4.*np.sqrt(6./(n_visible + n_hidden)),
        size=(n_visible, n_hidden))
    W = theano.shared(
        np.asarray(w_init, dtype=theano.config.floatX),
        name='W',
        borrow=True)
    b_hidden = theano.shared(
        np.zeros((n_hidden,), dtype=theano.config.floatX),
        name='b_hidden',
        borrow=True
    )
    b_visible = theano.shared(
        np.zeros((n_visible,), dtype=theano.config.floatX),
        name='b_visible',
        borrow=True
    )

    hidden_output = T.nnet.sigmoid(T.dot(input, W)+b_hidden)
    reconstructed = T.nnet.sigmoid(T.dot(hidden_output, W.T)+b_visible)
    return reconstructed, [W, b_hidden, b_visible]


def run_3_denoising_autoencoder(corruption_level=0.3):
    mnist_pkl = get_dataset('mnist')
    with open(mnist_pkl) as f:
        train_set, _, _ = pickle.load(f)

    batch_size = 20
    learning_rate = 0.1
    training_epochs = 15
    n_in=28*28
    n_hidden=500
    rng = np.random.RandomState(1234)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    train_set_x, train_set_y = load_dataset(train_set)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size

    x = T.matrix('x')

    corrupted_input = theano_rng.binomial(size=x.shape, n=1, p=1-corruption_level, dtype=theano.config.floatX)*x
    reconstructed, params = denoising_autoencoder(corrupted_input, n_in, n_hidden, rng)
    cost = mean_cross_entropy(reconstructed, x)

    minibatch_index = T.iscalar('minibatch_index')
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
        for minibatch_index in range(n_train_batches):
            costs.append(train_model(minibatch_index))
        print("Mean costs at epoch %d is %f%%" % (epoch, np.mean(costs)))

    total_time = time.time()-start_time
    print('The training code run %.1fs, for %d epochs, for with %f epochs/sec' % (total_time, epoch, epoch/total_time))

    filters = tile_raster_images(X=params[0].get_value(borrow=True).T,
                                 img_shape=(28, 28), tile_shape=(10, 10),
                                 tile_spacing=(1, 1))
    filters = cv2.resize(filters, dsize=None, fx=2., fy=2.)
    cv2.imshow('filters', filters)
    cv2.waitKey(-1)


if __name__ == "__main__":
    run_3_denoising_autoencoder(corruption_level=0.0)
