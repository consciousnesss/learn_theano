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


def one_zero_loss(prediction_labels, labels):
    return T.mean(T.neq(prediction_labels, labels))


def negative_log_likelihood_loss(prediction_probailities, labels):
    return -T.mean(T.log(prediction_probailities)[T.arange(labels.shape[0]), labels])


def mean_cross_entropy(prediction_probailities, labels):
    return -T.mean(T.sum(labels*T.log(prediction_probailities) + (1-labels)*T.log(1-prediction_probailities), axis=1))


def load_dataset(dataset):
    set_x = theano.shared(np.asarray(dataset[0], dtype=theano.config.floatX), borrow=True)
    set_y = theano.shared(np.asarray(dataset[1], dtype=theano.config.floatX), borrow=True)
    return set_x, T.cast(set_y, 'int32')


class RBM(object):
    def __init__(self, n_visible, n_hidden, rng):
        w_init = rng.uniform(
            low=-4*np.sqrt(6./(n_visible + n_hidden)),
            high=4*np.sqrt(6./(n_visible + n_hidden)),
            size=(n_visible, n_hidden))
        self.W = theano.shared(
            np.asarray(w_init, dtype=theano.config.floatX),
            name='W',
            borrow=True)
        self.b_hidden = theano.shared(
            np.zeros((n_hidden,), dtype=theano.config.floatX),
            name='b_hidden',
            borrow=True
        )

        self.b_visible = theano.shared(
            np.zeros((n_visible,), dtype=theano.config.floatX),
            name='b_visible',
            borrow=True
        )

    def sample_hidden_given_visible(self, visible, theano_rng):
        hidden_linear_activations = T.dot(visible, self.W)+self.b_hidden
        hidden_activation = T.nnet.sigmoid(hidden_linear_activations)
        hidden_sample = theano_rng.binomial(
            size=hidden_activation.shape, n=1, p=hidden_activation, dtype=theano.config.floatX)
        return hidden_sample, hidden_activation, hidden_linear_activations

    def sample_visible_give_hidden(self, hidden, theano_rng):
        linear_visible_activations = T.dot(hidden, self.W.T)+self.b_visible
        visible_activation = T.nnet.sigmoid(linear_visible_activations)
        visible_sample = theano_rng.binomial(
            size=visible_activation.shape, n=1, p=visible_activation, dtype=theano.config.floatX)
        return visible_sample, visible_activation, linear_visible_activations

    def gibbs_update_hidden_visible_hidden(self, hidden, theano_rng):
        visible_sample, visible_activation, linear_visible_activation = self.sample_visible_give_hidden(hidden, theano_rng)
        hidden_sample, hidden_activation, hidden_linear_activation = self.sample_hidden_given_visible(visible_sample, theano_rng)
        return [visible_sample, visible_activation, linear_visible_activation,
                hidden_sample, hidden_activation, hidden_linear_activation]

    def gibbs_update_visible_hidden_visible(self, visible, theano_rng):
        hidden_sample, hidden_activation, hidden_linear_activation = self.sample_hidden_given_visible(visible, theano_rng)
        visible_sample, visible_activation, linear_visible_activation = self.sample_visible_give_hidden(hidden_sample, theano_rng)
        return [hidden_sample, hidden_activation, hidden_linear_activation,
                visible_sample, visible_activation, linear_visible_activation]

    def free_energy(self, visible_sample):
        visible_bias_term = T.dot(visible_sample, self.b_visible)

        wx_b = T.dot(visible_sample, self.W) + self.b_hidden
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return T.mean(-hidden_term - visible_bias_term)

    def get_cost_updates(self, input, learning_rate, number_of_gibbs_steps, theano_rng, persistent_state):
        if persistent_state is None:
            chain_start = self.sample_hidden_given_visible(input, theano_rng)
        else:
            chain_start = persistent_state
        (visible_samples, visible_activations, linear_visible_activations,
         hidden_samples, hidden_activations, hidden_linear_activations), updates = theano.scan(
            fn=lambda x: self.gibbs_update_hidden_visible_hidden(x, theano_rng),
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=number_of_gibbs_steps
        )
        visible_chain_end = visible_samples[-1]
        cost = self.free_energy(input) - self.free_energy(visible_chain_end)

        parameters = [self.W, self.b_hidden, self.b_visible]
        for p in parameters:
            gradient = T.grad(cost, p, consider_constant=[visible_chain_end])
            updates[p] = p - T.cast(learning_rate, theano.config.floatX)*gradient

        if persistent_state is None:
            # here we use linear_visible_activations because theano can not make log(scan(sigm(..))) stable,
            # but it can make log(sigm(..)) stable.
            monitoring_cost = mean_cross_entropy(T.nnet.sigmoid(linear_visible_activations[-1]), input)
        else:
            updates[persistent_state] = hidden_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(input, updates)

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, input, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(input)
        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        n_visible = self.b_visible.get_value(borrow=True).shape[0]
        cost = T.mean(n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % n_visible

        return cost



def run_5_rbm():
    mnist_pkl = get_dataset('mnist')
    with open(mnist_pkl) as f:
        train_set, valid_set, test_set = pickle.load(f)

    batch_size = 20
    learning_rate = 0.1
    n_training_epochs = 1 #15
    n_visible=28*28
    n_hidden=500
    n_contrastive_divergence_steps=15

    # for sampling from trained model
    n_chains = 20
    n_samples = 2
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    train_set_x, _ = load_dataset(train_set)
    test_set_x, _ = load_dataset(test_set)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size

    x = T.matrix('x')

    persistent_chain = theano.shared(
        np.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)

    rbm = RBM(n_visible, n_hidden, rng)

    #persistent contrastive divergence with n_contrastive_divergence_steps steps
    cost, updates = rbm.get_cost_updates(
        x, learning_rate, number_of_gibbs_steps=n_contrastive_divergence_steps,
        theano_rng=theano_rng, persistent_state=persistent_chain)

    minibatch_index = T.iscalar('minibatch_index')

    train_rbm = theano.function(
        inputs=[minibatch_index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
        }
    )

    start_time = time.time()

    for epoch in range(n_training_epochs):
        epoch_start_time = time.time()
        costs = []
        for batch_index in range(n_train_batches):
            batch_start_time = time.time()
            costs.append(train_rbm(batch_index))
            print('Done batch %d of %d. Took %.4fs' % (batch_index, n_train_batches, time.time() - batch_start_time))
        print('Training epoch %d of %d, cost is %f, took %.1fs' %
              (epoch, n_training_epochs, np.mean(costs), time.time() - epoch_start_time))
        filters = tile_raster_images(X=rbm.W.get_value(borrow=True).T, img_shape=(28, 28))
        cv2.imshow('filter', filters)
        cv2.waitKey(-1)

    print ('Training took %d minutes' % (time.time()-start_time/60.))

    # sample from trained RBM
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        np.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )

    plot_every = 1000
    (hidden_samples, hidden_activations, hidden_linear_activations,
     visible_samples, visible_activations, linear_visible_activations), sampling_updates = theano.scan(
        fn=lambda x: rbm.gibbs_update_visible_hidden_visible(x, theano_rng),
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    sampling_updates[persistent_vis_chain] = visible_samples[-1]
    sample_fn = theano.function(
        [],
        [
            visible_activations[-1],
            visible_samples[-1]
        ],
        updates=sampling_updates
    )

    image_data = np.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )
    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    cv2.imshow('sampling', image_data)
    cv2.waitKey(-1)



if __name__ == "__main__":
    run_5_rbm()
