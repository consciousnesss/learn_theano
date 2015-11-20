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


def sigmoid_layer(input, n_in, n_out, rng):
    w_init = rng.uniform(
        low=-4*np.sqrt(6./(n_in + n_out)),
        high=4*np.sqrt(6./(n_in + n_out)),
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


def deep_mlp(input, n_in, n_out, hidden_layers_sizes, rng):
    current_input = input
    current_input_size = n_in
    params = []
    layers_description = []
    for i, hidden_layer_size in enumerate(hidden_layers_sizes):
        layer_output, layer_params = sigmoid_layer(current_input, current_input_size, hidden_layer_size, rng)
        params += layer_params
        layers_description.append((current_input, layer_params))
        current_input = layer_output
        current_input_size = hidden_layer_size
    output_layer, output_layer_params = logistic_layer(current_input, current_input_size, n_out)
    params += output_layer_params
    return output_layer, params, layers_description


class RBM(object):
    def __init__(self, W, b_hidden):
        self.W = W
        self.b_hidden = b_hidden
        n_visible = W.get_value(borrow=True).shape[0]
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
            chain_start, _, _ = self.sample_hidden_given_visible(input, theano_rng)
        else:
            chain_start = persistent_state
        (visible_samples, visible_activations, linear_visible_activations,
         hidden_samples, hidden_activations, hidden_linear_activations), updates = theano.scan(
            fn=lambda x: self.gibbs_update_hidden_visible_hidden(x, theano_rng),
            outputs_info=[None, None, None, chain_start, None, None],
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
        cost = T.mean(n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % n_visible

        return cost


def train_dbn():
    batch_size = 10
    finetune_learning_rate = 0.1
    pretrain_learning_rate = 0.01
    n_pretraining_epochs = 100
    n_finetune_training_epochs = 100
    n_in=28*28
    n_out=10
    hidden_layers_sizes=[1000, 1000, 1000]
    n_contrastive_divergence_steps=1
    persistent_contrastive_divergence=True

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    train_set, valid_set, test_set = get_dataset('mnist')
    train_set_x, train_set_y = load_dataset(train_set)
    valid_set_x, valid_set_y = load_dataset(valid_set)
    test_set_x, test_set_y = load_dataset(test_set)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size

    test_batch_size = valid_set_x.get_value(borrow=True).shape[0]
    n_validation_batches = valid_set_x.get_value(borrow=True).shape[0]/test_batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]/test_batch_size

    x = T.matrix('x')
    y = T.ivector('y')

    minibatch_index = T.iscalar('minibatch_index')

    mlp_output, mlp_params, mlp_layers_description = deep_mlp(x, n_in, n_out, hidden_layers_sizes, rng)
    pretrain_functions = []
    for layer_input, (W, b) in mlp_layers_description:
        rbm_layer = RBM(W=W, b_hidden=b)

        if persistent_contrastive_divergence:
            n_hidden = b.get_value(borrow=True).shape[0]
            persistent_chain = theano.shared(
                np.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)
        else:
            persistent_chain = None

        layer_cost, layer_updates = rbm_layer.get_cost_updates(
            layer_input, pretrain_learning_rate, n_contrastive_divergence_steps, theano_rng, persistent_chain)

        pretrain_rbm = theano.function(
            inputs=[minibatch_index],
            outputs=layer_cost,
            updates=layer_updates,
            givens={
                x: train_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
            }
        )
        pretrain_functions.append(pretrain_rbm)

    # PRETRAINING
    start_time = time.time()
    for i, pretrain_function in enumerate(pretrain_functions):
        layer_start_time = time.time()
        for epoch in range(n_pretraining_epochs):
            epoch_start_time = time.time()
            costs = []
            for batch_index in range(n_train_batches):
                costs.append(pretrain_function(batch_index))
            print('Training epoch %d of %d, cost is %f, took %.1fs' %
                  (epoch, n_pretraining_epochs, np.mean(costs), time.time() - epoch_start_time))
        print('Pretraining of layer %d took %d min' % (i, (time.time()-layer_start_time)/60.))

    print ('Pre training took %d minutes' % ((time.time()-start_time)/60.))

    # FINETUNING
    y_predict = T.argmax(mlp_output, axis=1)
    finetune_cost = negative_log_likelihood_loss(mlp_output, y)

    finetune_train_model = theano.function(
        inputs=[minibatch_index],
        outputs=[finetune_cost],
        updates=[[p, p - finetune_learning_rate*T.grad(finetune_cost, p)]
                 for p in mlp_params],
        givens={
            x: train_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
            y: train_set_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
        },
        profile=False
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
        patience = 4 * n_train_batches
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = min(n_train_batches, patience / 2)
        test_score = 0.
        best_validation_loss = np.inf

        print('Going to run the finetuning training with floatX=%s' % (theano.config.floatX))
        for epoch in range(n_finetune_training_epochs):
            epoch_start = time.time()
            for minibatch_index_value in range(n_train_batches):
                finetune_train_model(minibatch_index_value)
                iteration = epoch*n_train_batches + minibatch_index_value
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
            print(' - finished epoch %d out of %d in %.1fs' %
                  (epoch, n_finetune_training_epochs, time.time() - epoch_start))
        return epoch, best_validation_loss, test_score

    epoch, best_validation_loss, test_score = main_loop()

    total_time = time.time()-start_time
    print('Optimization complete in %d min with best validation score of %f %%, with test performance %f %%' %
          (total_time/60, best_validation_loss * 100., test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, epoch/total_time))


if __name__ == "__main__":
    train_dbn()
    '''
    ---No pretraining:
    Optimization complete in 18 min with best validation score of 1.760000 %, with test performance 1.940000 %
The code run for 65 epochs, with 0.059589 epochs/sec

    ---Pretraining for 10 epochs:
    Pre training took 12 minutes
    Optimization complete in 27 min with best validation score of 1.680000 %, with test performance 1.590000 %
The code run for 99 epochs, with 0.060696 epochs/sec

    ---Pretraining for 30 epochs:
    Pre training took 36 minutes
    Optimization complete in 26 min with best validation score of 1.500000 %, with test performance 1.450000 %
The code run for 99 epochs, with 0.061458 epochs/sec

    ---Pretraining for 100 epochs:
    Pre training took 125 minutes
    Optimization complete in 26 min with best validation score of 1.390000 %, with test performance 1.430000 %
The code run for 99 epochs, with 0.063411 epochs/sec

    It seems important to train the first layer really well because other layers are going to base their model on it.
    '''
