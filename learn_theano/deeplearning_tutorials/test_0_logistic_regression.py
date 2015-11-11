
import theano
import theano.tensor as T
import numpy as np
from learn_theano.utils.download_all_datasets import get_dataset
import cPickle
import time


def one_zero_loss(prediction_labels, labels):
    return T.mean(T.neq(prediction_labels, labels))


def negative_log_likelihood_loss(prediction_probailities, labels):
    return -T.mean(T.log(prediction_probailities)[T.arange(labels.shape[0]), labels])


def load_dataset(dataset):
    set_x = theano.shared(np.asarray(dataset[0], dtype=theano.config.floatX), borrow=True)
    set_y = theano.shared(np.asarray(dataset[1], dtype=theano.config.floatX), borrow=True)
    return set_x, T.cast(set_y, 'int32')


def run_0_logistic_regression():
    mnist_pkl = get_dataset('mnist')
    with open(mnist_pkl) as f:
        train_set, valid_set, test_set = cPickle.load(f)

    batch_size = 600
    learning_rate = 0.13
    n_epochs = 1000

    train_set_x, train_set_y = load_dataset(train_set)
    valid_set_x, valid_set_y = load_dataset(valid_set)
    test_set_x, test_set_y = load_dataset(test_set)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_validation_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size

    x = T.matrix('x')
    y = T.ivector('y')

    n_in=28*28
    n_out=10

    W = theano.shared(
        np.zeros((n_in, n_out), dtype=theano.config.floatX),
        name='W',
        borrow=True)
    b = theano.shared(
        np.zeros((n_out,), dtype=theano.config.floatX),
        name='b',
        borrow=True
    )
    py_given_x = T.nnet.softmax(T.dot(x, W)+b)
    y_predict = T.argmax(py_given_x, axis=1)

    cost = negative_log_likelihood_loss(py_given_x, y)

    minibatch_index = T.iscalar('minibatch_index')

    train_model = theano.function(
        inputs=[minibatch_index],
        outputs=cost,
        updates=(
            [W, W - learning_rate*T.grad(cost, W)],
            [b, b - learning_rate*T.grad(cost, b)],
        ),
        givens={
            x: train_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
            y: train_set_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
        }
    )

    validation_model = theano.function(
        inputs=[minibatch_index],
        outputs=one_zero_loss(y_predict, y),
        givens={
            x: valid_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
            y: valid_set_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
        }
    )

    test_model = theano.function(
        inputs=[minibatch_index],
        outputs=one_zero_loss(y_predict, y),
        givens={
            x: test_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
            y: test_set_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
        }
    )

    start_time = time.time()

    def main_loop():
        patience = 5000
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = n_train_batches
        test_score = 0.
        best_validation_loss = np.inf
        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                batch_cost = train_model(minibatch_index)

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
    run_0_logistic_regression()
