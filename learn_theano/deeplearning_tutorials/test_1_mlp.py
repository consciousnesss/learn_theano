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


def hidden_layer(input, n_in, n_out, rng):
    w_init = rng.uniform(
        low=-np.sqrt(6./(n_in + n_out)),
        high=np.sqrt(6./(n_in + n_out)),
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
    return T.tanh(T.dot(input, W)+b), [W, b]


def run_1_mlp():
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

    train_model_impl = theano.function(
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

    theano.printing.pydotprint(learning_rate*T.grad(cost, hidden_layer_params[0]),
                               outfile="logreg_pydotprint_prediction.png", var_with_name_simple=True)

    def train_model(*args):
        return train_model_impl(*args)

    validation_model_impl = theano.function(
        inputs=[minibatch_index],
        outputs=one_zero_loss(y_predict, y),
        givens={
            x: valid_set_x[minibatch_index*test_batch_size:(minibatch_index+1)*test_batch_size],
            y: valid_set_y[minibatch_index*test_batch_size:(minibatch_index+1)*test_batch_size],
        }
    )

    def validation_model(*args):
        return validation_model_impl(*args)

    test_model_impl = theano.function(
        inputs=[minibatch_index],
        outputs=one_zero_loss(y_predict, y),
        givens={
            x: test_set_x[minibatch_index*test_batch_size:(minibatch_index+1)*test_batch_size],
            y: test_set_y[minibatch_index*test_batch_size:(minibatch_index+1)*test_batch_size],
        }
    )

    def test_model(*args):
        return test_model_impl(*args)

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
    '''
    Expected results:
    cpu:
    Optimization complete in 2131.8s with best validation score of 1.680000 %, with test performance 1.650000 %
The code run for 999 epochs, with 0.468623 epochs/sec
    '''
    run_1_mlp()
    # from learn_theano.utils.profiler_run import profiler_run
    # profiler_run('run_1_mlp()',
    #              print_callers_of="function_module.py:482")
