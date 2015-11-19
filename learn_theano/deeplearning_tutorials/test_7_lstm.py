from learn_theano.utils.download_all_datasets import get_dataset
import numpy as np
import theano
import time
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

theano.config.floatX = 'float32'


def dropout_layer(input, use_noise, theano_rng):
    noise = theano_rng.binomial(input.shape, p=0.5, n=1, dtype=input.dtype)
    proj = T.switch(use_noise, input*noise, input*0.5)
    return proj


def lstm_layer(input, input_mask, W_lstm, U_lstm, b_lstm, dim_proj):
    '''
    Input shape is  [max_length_of_sequence, number_of_sequences, number_of_features]
    Mask shape is [max_length_of_sequence, number_of_sequences]. It is 1 where sequence item is valid
    and zero where it was padded.
    '''
    nsteps = input.shape[0]
    n_samples = input.shape[1]

    def _slice(_x, n, dim):
        assert(_x.ndim == 2)
        return _x[:, n*dim: (n+1)*dim]

    def _step(input_pre_activation, sample_mask, previous_cell_output_h, previous_cell_state):
        preact = T.dot(previous_cell_output_h, U_lstm) + input_pre_activation

        input_gate_activation = T.nnet.sigmoid(_slice(preact, 0, dim_proj))
        forget_gate_activation = T.nnet.sigmoid(_slice(preact, 1, dim_proj))
        output_gate_actiavation = T.nnet.sigmoid(_slice(preact, 2, dim_proj))
        candidate_cell_state = T.tanh(_slice(preact, 3, dim_proj))

        new_cell_state = forget_gate_activation * previous_cell_state + input_gate_activation * candidate_cell_state
        # this preserves the state of the cells for the samples that has finished before the max length
        new_cell_state = sample_mask[:, None] * new_cell_state + (1. - sample_mask)[:, None] * previous_cell_state

        cell_output_h = output_gate_actiavation * T.tanh(new_cell_state)
        cell_output_h = sample_mask[:, None] * cell_output_h + (1. - sample_mask)[:, None] * previous_cell_output_h

        return cell_output_h, new_cell_state

    # precompute activations for all the inputs
    input_pre_activations = (T.dot(input, W_lstm) + b_lstm)

    rval, updates = theano.scan(_step,
                                sequences=[input_pre_activations, input_mask],
                                outputs_info=[T.alloc(np.asarray(0., dtype=theano.config.floatX), n_samples, dim_proj),
                                              T.alloc(np.asarray(0., dtype=theano.config.floatX), n_samples, dim_proj)],
                                name='lstm_layers',
                                n_steps=nsteps)
    assert(len(updates) == 0)
    return rval[0]


def build_model(W_embedding, W_lstm, U_lstm, b_lstm, W_classifier, b_classifier, dim_proj, use_dropout):
    theano_rng = RandomStreams(123)

    # Used for dropout.
    use_noise = theano.shared(np.asarray(0., dtype=theano.config.floatX))

    # matrix of word indices with (number
    x = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=theano.config.floatX)
    y = T.vector('y', dtype='int64')

    n_timesteps = x.shape[0]  # max length of the sequence for this batch
    n_samples = x.shape[1]  # number of sequences (equals to batch size usually)

    embeddings = W_embedding[x.flatten()].reshape([n_timesteps, n_samples, dim_proj])
    lstm_output = lstm_layer(embeddings, mask, W_lstm, U_lstm, b_lstm, dim_proj)

    mean_pooling = (lstm_output * mask[:, :, None]).sum(axis=0)
    mean_pooling = mean_pooling / mask.sum(axis=0)[:, None]
    if use_dropout:
        mean_pooling = dropout_layer(mean_pooling, use_noise, theano_rng)

    prediction = T.nnet.softmax(T.dot(mean_pooling, W_classifier) + b_classifier)

    f_pred_prob = theano.function([x, mask], prediction, name='f_pred_prob')
    f_pred = theano.function([x, mask], prediction.argmax(axis=1), name='f_pred')

    cost = -T.log(prediction[T.arange(n_samples), y] + 1e-8).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost


def ortho_weight(ndim):
    '''
    Orthogonal weights matrix. Eigenvalues of matrix are all 1s. Based on http://arxiv.org/abs/1312.6120
    '''
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


def init_params(n_words, dim_proj, ydim):
    randn = np.random.rand(n_words, dim_proj)
    W_embedding = theano.shared(
        (0.01 * randn).astype(theano.config.floatX),
        name='W_embedding',
        borrow=True)

    # concatenate weights for all LSTM elements: input gates, output gates, forgetting gates and hypothesis
    W_lstm = theano.shared(
        np.concatenate([ortho_weight(dim_proj) for _ in range(4)], axis=1),
        name='W_lstm',
        borrow=True)
    U_lstm = theano.shared(
        np.concatenate([ortho_weight(dim_proj) for _ in range(4)], axis=1),
        name='U_lstm',
        borrow=True)
    b_lstm = theano.shared(
        np.zeros((4 * dim_proj,), dtype=theano.config.floatX),
        name='b_lstm',
        borrow=True)

    # classifier
    W_classifier = theano.shared(
        0.01 * np.random.randn(dim_proj, ydim).astype(theano.config.floatX),
        name='W_classifier',
        borrow=True)
    b_classifier = theano.shared(
        np.zeros((ydim,)).astype(theano.config.floatX),
        name='b_classifier',
        borrow=True)

    return W_embedding, W_lstm, U_lstm, b_lstm, W_classifier, b_classifier


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % p.name) for p in tparams]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, name='sgd_f_grad_shared')
    parameter_updates = [(p, p - lr * g) for p, g in zip(tparams, gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=parameter_updates, name='sgd_f_update')
    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * np.asarray(0., dtype=theano.config.floatX),
                                  name='%s_grad' % p.name)
                    for p in tparams]
    running_up2 = [theano.shared(p.get_value() * np.asarray(0., dtype=theano.config.floatX),
                                 name='%s_rup2' % p.name)
                   for p in tparams]
    running_grads2 = [theano.shared(p.get_value() * np.asarray(0., dtype=theano.config.floatX),
                                    name='%s_rgrad2' % p.name)
                      for p in tparams]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams, updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n)
    if shuffle:
        np.random.shuffle(idx_list)
    return [idx_list[i*minibatch_size:(i+1)*minibatch_size] for i in range(n/minibatch_size)]


def prepare_data(seqs):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)

    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask


def pred_error(f_pred, data, minibatch_indices):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    """
    valid_err = 0
    for valid_index in minibatch_indices:
        x, mask = prepare_data(np.array(data[0])[valid_index])
        preds = f_pred(x, mask)
        targets = np.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - np.asarray(valid_err, dtype=theano.config.floatX) / len(data[0])

    return valid_err


def run_7_lstm_training():
    train_set, valid_set, test_set = get_dataset('imdb')

    ydim = 2  # n_out
    dim_proj = 128
    n_words = 10000  # this is implied in preprocessed imdb dataset
    use_dropout = True
    optimizer = adadelta
    valid_batch_size=64
    validFreq=370
    max_epochs= 100
    batch_size=16
    lrate=0.0001
    dispFreq=10
    patience=10

    W_embedding, W_lstm, U_lstm, b_lstm, W_classifier, b_classifier = init_params(n_words, dim_proj, ydim)

    # use_noise is for dropout
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(
        W_embedding, W_lstm, U_lstm, b_lstm, W_classifier, b_classifier, dim_proj, use_dropout)

    grads = T.grad(cost, wrt=[W_embedding, W_lstm, U_lstm, b_lstm, W_classifier, b_classifier])

    lr = T.scalar(name='learning_rate')
    f_grad_shared, f_update = optimizer(lr, [W_embedding, W_lstm, U_lstm, b_lstm, W_classifier, b_classifier],
                                        grads, x, mask, y, cost)

    print 'Optimization'

    validation_minibatched_inidices = get_minibatches_idx(len(valid_set[0]), valid_batch_size)
    test_minibatched_inidices = get_minibatches_idx(len(test_set[0]), valid_batch_size)

    history_errs = []
    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            training_minibatches_indices = get_minibatches_idx(len(train_set[0]), batch_size, shuffle=True)

            for train_index in training_minibatches_indices:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                current_x = [train_set[0][t] for t in train_index]
                current_y = [train_set[1][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                current_x, current_mask = prepare_data(current_x)
                n_samples += current_x.shape[1]

                cost = f_grad_shared(current_x, current_mask, current_y)
                f_update(lrate)

                if uidx % dispFreq == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if uidx % validFreq == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, train_set, training_minibatches_indices)
                    valid_err = pred_error(f_pred, valid_set, validation_minibatched_inidices)
                    test_err = pred_error(f_pred, test_set, test_minibatched_inidices)

                    history_errs.append([valid_err, test_err])

                    if (valid_err <= np.array(history_errs)[:, 0].min()):
                        bad_counter = 0

                    print ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err)

                    if (len(history_errs) > patience and
                            valid_err >= np.array(history_errs)[:-patience, 0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    use_noise.set_value(0.)
    train_minibatch_indices_sorted = get_minibatches_idx(len(train_set[0]), batch_size)
    train_err = pred_error(f_pred, train_set, train_minibatch_indices_sorted)
    valid_err = pred_error(f_pred, valid_set, validation_minibatched_inidices)
    test_err = pred_error(f_pred, test_set, test_minibatched_inidices)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print ('Training took %.1fs' % (end_time - start_time))
    return train_err, valid_err, test_err


if __name__ == '__main__':
    run_7_lstm_training()

    '''
    Results:
    Epoch  69 Update  8880 Cost  4.28049634138e-06
('Train ', 0.0, 'Valid ', 0.57943925233644866, 'Test ', 0.20799999999999996)
Early Stop!
Seen 768 samples
Train  0.0 Valid  0.579439252336 Test  0.208
The code run for 70 epochs, with 14.774699 sec/epochs
Training took 1034.2s

    Lessons:
     - there are non-trivial optimizers such as ADADELTA that perform better than SGD
     - there are complications if you use datasets with varying sample size: you need to pad etc.
     - orthogonal weights initialization
     - randomizing batches on each epoch
     - concatenate all weights of LSTM cells
    '''
