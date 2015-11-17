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


def autoencoder(input, W, b_hidden):
    b_visible = theano.shared(
        np.zeros((W.get_value(borrow=True).shape[0],), dtype=theano.config.floatX),
        name='b_visible',
        borrow=True
    )
    hidden_output = T.nnet.sigmoid(T.dot(input, W)+b_hidden)
    reconstructed = T.nnet.sigmoid(T.dot(hidden_output, W.T)+b_visible)
    return reconstructed, [W, b_hidden, b_visible]


def run_4_stacked_autoencoder():
    mnist_pkl = get_dataset('mnist')
    with open(mnist_pkl) as f:
        train_set, valid_set, test_set = pickle.load(f)

    batch_size = 1
    finetune_learning_rate = 0.1
    finetune_training_epochs = 50

    pretrain_learning_rate = 0.001
    pretraining_epochs = 15
    n_in=28*28
    hidden_layers_sizes=[1000, 1000, 1000]
    corruption_levels = [.1, .2, .3]
    n_out=10
    rng = np.random.RandomState(89677)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    train_set_x, train_set_y = load_dataset(train_set)
    valid_set_x, valid_set_y = load_dataset(valid_set)
    test_set_x, test_set_y = load_dataset(test_set)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size

    test_batch_size = valid_set_x.get_value(borrow=True).shape[0]
    n_validation_batches = valid_set_x.get_value(borrow=True).shape[0]/test_batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]/test_batch_size

    # construct deep mlp
    x = T.matrix('x')
    y = T.ivector('y')
    mlp_output, mlp_params, layers_description = deep_mlp(x, n_in=n_in, n_out=n_out, hidden_layers_sizes=hidden_layers_sizes, rng=rng)

    minibatch_index = T.iscalar('minibatch_index')

    # pretrain
    pretraining_models = []
    for i, (layer_input, (W, b_hidden)) in enumerate(layers_description):
        corrupted_input = theano_rng.binomial(
            size=layer_input.shape, n=1, p=1-corruption_levels[i], dtype=theano.config.floatX)*layer_input
        reconstructed_output, autoencoder_params = autoencoder(corrupted_input, W, b_hidden)
        pretraining_cost = mean_cross_entropy(reconstructed_output, layer_input)
        pretraining_model = theano.function(
            inputs=[minibatch_index],
            outputs=[pretraining_cost],
            updates=[[p, p - pretrain_learning_rate*T.grad(pretraining_cost, p)]
                     for p in autoencoder_params],
            givens={
                x: train_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
            }
        )
        pretraining_models.append(pretraining_model)

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

    for i, pretraining_model in enumerate(pretraining_models):
        pretraining_start_time = time.time()

        print('Going to run the pretraining for layer %d with floatX=%s' % (i, theano.config.floatX))
        for epoch in range(pretraining_epochs):
            costs = []
            epoch_start_time = time.time()
            for minibatch_index in range(n_train_batches):
                costs.append(pretraining_model(minibatch_index))
            print("Layer %d: mean costs at epoch %d is %f%% (ran for %.1fs)" %
                  (i, epoch, np.mean(costs), time.time() - epoch_start_time))

        total_pretraining_time = time.time()-pretraining_start_time
        print('The pretraining code for layer %d run %.1fs, for %d epochs, for with %f epochs/sec' %
              (i, total_pretraining_time, epoch, epoch/total_pretraining_time))

    start_time = time.time()

    def main_loop():
        patience = 10 * n_train_batches
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = min(n_train_batches, patience / 2)
        test_score = 0.
        best_validation_loss = np.inf

        print('Going to run the finetuning training with floatX=%s' % (theano.config.floatX))
        for epoch in range(finetune_training_epochs):
            epoch_start = time.time()
            for minibatch_index in range(n_train_batches):
                finetune_train_model(minibatch_index)
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
            print(' - finished epoch %d out of %d in %.1fs' %
                  (epoch, finetune_training_epochs, time.time() - epoch_start))
        return epoch, best_validation_loss, test_score

    epoch, best_validation_loss, test_score = main_loop()

    total_time = time.time()-start_time
    print('Optimization complete in %.1fs with best validation score of %f %%, with test performance %f %%' %
          (total_time, best_validation_loss * 100., test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, epoch/total_time))


if __name__ == "__main__":
    run_4_stacked_autoencoder()
    '''
    No pretraining:
Optimization complete in 1478.6s with best validation score of 1.960000 %, with test performance 2.130000 %
The code run for 15 epochs, with 0.010145 epochs/sec

    With pretraining:
The pretraining code for layer 0 run 676.4s, for 14 epochs, for with 0.020697 epochs/sec
The pretraining code for layer 1 run 902.2s, for 14 epochs, for with 0.015518 epochs/sec
The pretraining code for layer 2 run 1080.3s, for 14 epochs, for with 0.012960 epochs/sec

    Optimization complete in 4583.6s with best validation score of 1.450000 %, with test performance 1.360000 %
The code run for 49 epochs, with 0.010690 epochs/se
    '''
