#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy as np
from learn_theano.utils.download_all_datasets import get_dataset
import cPickle
import pickle
import time


def one_zero_loss(prediction_labels, labels):
    return T.mean(T.neq(prediction_labels, labels))


def negative_log_likelihood_loss(prediction_probailities, labels):
    return -T.mean(T.log(prediction_probailities)[T.arange(labels.shape[0]), labels])


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
            np.zeros((n_hidden,), dtype=theano.config.floatX),
            name='b_visible',
            borrow=True
        )

    def sample_hidden_given_visible(self, visible, theano_rng):
        hidden_activation = T.nnet.sigmoid(T.dot(visible, self.W)+self.b_hidden)
        hidden_sample = theano_rng.binominal(
            size=hidden_activation.shape, n=1, p=hidden_activation, dtype=theano.config.floatX)
        return hidden_sample

    def sample_visible_give_hidden(self, hidden, theano_rng):
        visible_activation = T.nnet.sigmoid(T.dot(hidden, self.W.T)+self.b_visible)
        visible_sample = theano_rng.binominal(
            size=visible_activation.shape, n=1, p=visible_activation, dtype=theano.config.floatX)
        return visible_sample

    def gibbs_update_hidden_visible_hidden(self, hidden, theano_rng):
        visible_sample = self.sample_visible_give_hidden(hidden, theano_rng)
        return visible_sample, self.sample_hidden_given_visible(visible_sample, theano_rng)

    def gibbs_update_visible_hidden_visible(self, visible, theano_rng):
        hidden_sample = self.sample_hidden_given_visible(visible, theano_rng)
        return hidden_sample, self.sample_visible_give_hidden(hidden_sample, theano_rng)

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
        (visible_samples, hidden_samples), updates = theano.scan(
            fn=self.gibbs_update_hidden_visible_hidden,
            outputs_info=[None, chain_start],
            n_steps=number_of_gibbs_steps,
            non_sequences=theano_rng
        )
        visible_chain_end = visible_samples[-1]
        cost = self.free_energy(input) - self.free_energy(visible_chain_end)

        parameters = [self.W, self.b_hidden, self.b_visible]
        for p in parameters:
            updates[p] = p - T.cast(learning_rate, theano.config.floatX)



def run_5_rbm():
    mnist_pkl = get_dataset('mnist')
    with open(mnist_pkl) as f:
        train_set, valid_set, test_set = pickle.load(f)

    batch_size = 20
    learning_rate = 0.01
    n_epochs = 10
    L1_reg_coeff = 0.00
    L2_reg_coeff = 0.0001
    n_in=28*28
    n_hidden=500
    n_out=10
    rng = np.random.RandomState(1234)

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

    hidden_layer_output, hidden_layer_params = hidden_layer(x, n_in, n_hidden, rng)
    output_layer_output, output_layer_params = logistic_layer(hidden_layer_output, n_hidden, n_out)

    y_predict = T.argmax(output_layer_output, axis=1)

    # weights decay
    L1 = abs(hidden_layer_params[0]).sum() + abs(output_layer_params[0]).sum()
    L2 = T.sqr(hidden_layer_params[0]).sum() + T.sqr(output_layer_params[0]).sum()

    cost = negative_log_likelihood_loss(output_layer_output, y) + L1_reg_coeff*L1 + L2_reg_coeff*L2

    minibatch_index = T.iscalar('minibatch_index')

    train_model = theano.function(
        inputs=[minibatch_index],
        outputs=[],
        updates=[[p, p - learning_rate*T.grad(cost, p)]
                 for p in (output_layer_params + hidden_layer_params)],
        givens={
            x: train_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
            y: train_set_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
        },
        profile=True
    )

    validation_model = theano.function(
        inputs=[minibatch_index],
        outputs=one_zero_loss(y_predict, y),
        givens={
            x: valid_set_x[minibatch_index*test_batch_size:(minibatch_index+1)*test_batch_size],
            y: valid_set_y[minibatch_index*test_batch_size:(minibatch_index+1)*test_batch_size],
        }
    )

    test_model = theano.function(
        inputs=[minibatch_index],
        outputs=one_zero_loss(y_predict, y),
        givens={
            x: test_set_x[minibatch_index*test_batch_size:(minibatch_index+1)*test_batch_size],
            y: test_set_y[minibatch_index*test_batch_size:(minibatch_index+1)*test_batch_size],
        }
    )

    start_time = time.time()

    def main_loop():
        patience = 10000
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = n_train_batches
        test_score = 0.
        best_validation_loss = np.inf

        print('Going to run the training with floatX=%s' % (theano.config.floatX))
        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                train_model(minibatch_index)

                iteration = epoch*n_train_batches + minibatch_index
                if (iteration + 1) % validation_frequency == 0.:
                    validation_cost = np.mean([validation_model(i) for i in range(n_validation_batches)])
                    print('epoch %i, validation error %f %%' % (epoch, validation_cost * 100.))
                    if validation_cost < best_validation_loss:
                        if validation_cost < best_validation_loss*improvement_threshold:
                            patience = max(patience, iteration*patience_increase)
                        best_validation_loss = validation_cost
                        test_score = np.mean([test_model(i) for i in range(n_test_batches)])
                        print('  epoch %i, minibatch test error of best model %f %%' % (epoch, test_score * 100.))
                if patience <= iteration:
                    return epoch, best_validation_loss, test_score
        return epoch, best_validation_loss, test_score

    epoch, best_validation_loss, test_score = main_loop()

    total_time = time.time()-start_time
    print('Optimization complete in %.1fs with best validation score of %f %%, with test performance %f %%' %
          (total_time, best_validation_loss * 100., test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, epoch/total_time))


if __name__ == "__main__":
    run_5_rbm()
