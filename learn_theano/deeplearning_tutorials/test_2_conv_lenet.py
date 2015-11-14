#!/usr/bin/env python

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import numpy as np
from learn_theano.utils.download_all_datasets import get_dataset, get_3_wolves_image
import pickle
import time
import cv2
import matplotlib.pyplot as plots


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


def conv_layer(input, feature_maps_count_in, feature_maps_count_out, filter_shape, rng):
    weights_shape = (feature_maps_count_out, feature_maps_count_in, filter_shape[0], filter_shape[1])
    w_bound = np.sqrt(np.prod(weights_shape[1:]))

    w_init = rng.uniform(-1./w_bound, 1./w_bound, size=weights_shape)
    W = theano.shared(
        np.asarray(w_init, dtype=theano.config.floatX),
        name='W',
        borrow=True
    )
    b = theano.shared(
        np.zeros((feature_maps_count_out,), dtype=theano.config.floatX),
        name='b',
        borrow=True
    )
    return T.nnet.sigmoid(T.nnet.conv2d(input, W) + b.dimshuffle('x', 0, 'x', 'x')), [W, b]


def max_pooling_layer(input, maxpool_shape):
    return downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)


def conv_poll_layer(input, feature_maps_count_in, feature_maps_count_out, filter_shape, maxpool_shape, image_shape, rng):
    fan_in = feature_maps_count_in*np.prod(filter_shape)
    fan_out = feature_maps_count_out*np.prod(filter_shape)/np.prod(maxpool_shape)
    W_bound = np.sqrt(6. / (fan_in + fan_out))

    weights_shape = (feature_maps_count_out, feature_maps_count_in, filter_shape[0], filter_shape[1])
    W = theano.shared(
        np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=weights_shape),
                   dtype=theano.config.floatX),
        name='W',
        borrow=True
    )
    b = theano.shared(
        np.zeros((feature_maps_count_out,), dtype=theano.config.floatX),
        name='b',
        borrow=True
    )

    assert image_shape[1] == feature_maps_count_in
    convolution_out = T.nnet.conv2d(
        input,
        W,
        filter_shape=weights_shape,
        image_shape=image_shape  # this can be None, but is important optimization to make it 3 times faster
    )

    pooled_out = max_pooling_layer(convolution_out, maxpool_shape)
    return T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x')), [W, b]


def run_conv_net_image_filtering():
    img = get_3_wolves_image()
    img = img.astype(theano.config.floatX) / 256.

    prepared_image = img.transpose(2, 0, 1).reshape(1, 3, img.shape[0], img.shape[1])
    # now image shape is 1x3x639x516 - (batchsize x number_of_feature_maps x height x width)

    input = T.tensor4('input')
    rng = np.random.RandomState(23455)
    filter_symbolic, _ = conv_layer(
        input, feature_maps_count_in=3, feature_maps_count_out=2, filter_shape=(9, 9), rng=rng)
    filter = theano.function([input], filter_symbolic)

    filtered_image = filter(prepared_image)
    output0 = filtered_image[0, 0, :, :]
    output1 = filtered_image[0, 1, :, :]

    plots.subplot(1, 3, 1); plots.axis('off'); plots.imshow(img)
    plots.gray()
    plots.subplot(1, 3, 2); plots.axis('off'); plots.imshow(output0)
    plots.subplot(1, 3, 3); plots.axis('off'); plots.imshow(output1)
    plots.show()


def run_2_lenet_training():
    mnist_pkl = get_dataset('mnist')
    with open(mnist_pkl) as f:
        train_set, valid_set, test_set = pickle.load(f)

    batch_size = 500
    learning_rate = 0.1
    n_epochs = 200
    n_hidden = 500
    n_out=10
    rng = np.random.RandomState(23455)
    number_of_kernels = [20, 50]

    train_set_x, train_set_y = load_dataset(train_set)
    valid_set_x, valid_set_y = load_dataset(valid_set)
    test_set_x, test_set_y = load_dataset(test_set)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size

    test_batch_size = batch_size
    n_validation_batches = valid_set_x.get_value(borrow=True).shape[0]/test_batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]/test_batch_size

    x = T.matrix('x')
    y = T.ivector('y')

    layer_0_input = x.reshape((batch_size, 1, 28, 28))

    conv_layer_0_out, conv_layer_0_params = conv_poll_layer(
        layer_0_input,
        feature_maps_count_in=1,
        feature_maps_count_out=number_of_kernels[0],
        filter_shape=(5, 5),
        maxpool_shape=(2, 2),
        image_shape=(batch_size, 1, 28, 28),
        rng=rng
    )

    # filtering from the previous layer reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling from the prev. layer reduces this further to (24/2, 24/2) = (12, 12)
    conv_layer_1_out, conv_layer_1_params = conv_poll_layer(
        conv_layer_0_out,
        feature_maps_count_in=number_of_kernels[0],
        feature_maps_count_out=number_of_kernels[1],
        filter_shape=(5, 5),
        maxpool_shape=(2, 2),
        image_shape=(batch_size, number_of_kernels[0], 12, 12),
        rng=rng
    )

    hidden_layer_output, hidden_layer_params = hidden_layer(
        conv_layer_1_out.flatten(2),
        n_in=number_of_kernels[1]*4*4,  # 4 is the shape of output from layer1 maxpool layer
        n_out=n_hidden,
        rng=rng)

    output_layer_output, output_layer_params = logistic_layer(hidden_layer_output, n_hidden, n_out)

    y_predict = T.argmax(output_layer_output, axis=1)

    cost = negative_log_likelihood_loss(output_layer_output, y)

    minibatch_index = T.iscalar('minibatch_index')

    all_parameters = (conv_layer_0_params + conv_layer_1_params + hidden_layer_params + output_layer_params)

    train_model_impl = theano.function(
        inputs=[minibatch_index],
        outputs=[],
        updates=[[p, p - learning_rate*T.grad(cost, p)]
                 for p in (conv_layer_0_params + conv_layer_1_params + hidden_layer_params + output_layer_params)],
        givens={
            x: train_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
            y: train_set_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
        },
        profile=False
    )

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
                batch_start = time.time()
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

    with open('trained_lenet.pkl', 'w') as f:
        pickle.dump([p.get_value(borrow=True) for p in all_parameters], f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    run_2_lenet_training()
    '''
    Expected results:
    Optimization complete in 5445.7s (90min) with best validation score of 0.900000 %, with test performance 0.930000 %
    The code run for 199 epochs, with 0.036543 epochs/sec
    '''
