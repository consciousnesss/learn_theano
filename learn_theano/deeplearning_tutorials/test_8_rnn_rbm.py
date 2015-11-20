
import numpy as np
import theano
import os
import theano.tensor as T
from learn_theano.utils.play_midi import play_midi
from learn_theano.utils.download_all_datasets import get_dataset
from learn_theano.utils.midi.utils import midiread, midiwrite
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pylab as plots
import cPickle
import time


np.random.seed(0xbeef)
theano_rng = RandomStreams(seed=np.random.randint(1 << 30))


def build_rbm(v, W, bv, bh, k):
    '''Construct a k-step Gibbs chain starting at v for an RBM.

    v : Theano vector or matrix
        If a matrix, multiple chains will be run in parallel (batch).
    W : Theano matrix
        Weight matrix of the RBM.
    bv : Theano vector
        Visible bias vector of the RBM.
    bh : Theano vector
        Hidden bias vector of the RBM.
    k : scalar or Theano scalar
        Length of the Gibbs chain.

    Return a (v_sample, cost, monitor, updates) tuple:

    v_sample : Theano vector or matrix with the same shape as `v`
        Corresponds to the generated sample(s).
    cost : Theano scalar
        Expression whose gradient with respect to W, bv, bh is the CD-k
        approximation to the log-likelihood of `v` (training example) under the
        RBM. The cost is averaged in the batch case.
    monitor: Theano scalar
        Pseudo log-likelihood (also averaged in the batch case).
    updates: dictionary of Theano variable -> Theano variable
        The `updates` object returned by scan.'''

    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
        h = theano_rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                                dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
        v = theano_rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                                dtype=theano.config.floatX)
        return mean_v, v

    chain, updates = theano.scan(
        lambda v: gibbs_step(v)[1],
        outputs_info=[v],
        n_steps=k
    )
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates


def shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(np.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


def build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent, params):
    if params is None:
        W = shared_normal(n_visible, n_hidden, 0.01)
        bv = theano.shared(np.zeros((n_visible), dtype=theano.config.floatX))
        bh = theano.shared(np.zeros((n_hidden), dtype=theano.config.floatX))
        Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
        Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
        Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
        Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
        bu = theano.shared(np.zeros((n_hidden_recurrent), dtype=theano.config.floatX))
        params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu
    else:
        W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu = params

    v = T.matrix('v')
    # initial hidden values
    u0 = T.zeros((n_hidden_recurrent,))

    def recurrence(current_visible, current_hidden_u):
        # generate biases for RBM
        bv_t = bv + T.dot(current_hidden_u, Wuv)
        bh_t = bh + T.dot(current_hidden_u, Wuh)
        generate = current_visible is None
        if generate:
            current_visible, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t,
                                                     bh_t, k=1)
        hidden_u = T.tanh(bu + T.dot(current_visible, Wvu) + T.dot(current_hidden_u, Wuu))
        if generate:
            return ([current_visible, hidden_u], updates)
        else:
            return [hidden_u, bv_t, bh_t]

    (hidden_u, bv_t, bh_t), updates_train = theano.scan(
        lambda visible, previous_hidden, *args: recurrence(visible, previous_hidden),
        sequences=v,
        outputs_info=[u0, None, None],
        non_sequences=params,  # not clear why its needed,
    )

    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:], k=3)
    updates_train.update(updates_rbm)

    # function for generation
    (v_t, u_t), updates_generate = theano.scan(
        lambda u_tm1, *_: recurrence(None, u_tm1),
        outputs_info=[None, u0], non_sequences=params, n_steps=50)

    return (v, v_sample, cost, monitor, params, updates_train, v_t,
            updates_generate)


class RnnRBM(object):
    def __init__(self,
                 network_parameters,
                 n_hidden=500,
                 n_hidden_recurrent=500,
                 learning_rate=0.001,
                 pitch_range=(21, 109),
                 midi_sampling_period=0.3):
        '''
        :param pitch_range: Specifies the pitch range of the piano-roll in MIDI note numbers.
            The default (21, 109) corresponds to the full range of piano (88 notes).
        '''
        self._pitch_range = pitch_range
        self._midi_sampling_period = midi_sampling_period
        (v, v_sample, cost, monitor, params, updates_train, v_t, updates_generate) = build_rnnrbm(
            pitch_range[1] - pitch_range[0],
            n_hidden,
            n_hidden_recurrent,
            network_parameters
        )
        self._params = params

        gradient = T.grad(cost, params, consider_constant=[v_sample])
        updates_train.update(
            ((p, p - learning_rate * g) for p, g in zip(params, gradient))
        )
        self.train_function = theano.function(
            [v],
            monitor,
            updates=updates_train
        )
        self.generate_function = theano.function(
            [],
            v_t,
            updates=updates_generate
        )

    def train(self, files, batch_size, num_epochs):
        dataset = [midiread(f, self._pitch_range, self._midi_sampling_period).piano_roll.astype(theano.config.floatX)
                   for f in files]
        start_time = time.time()
        for epoch in xrange(num_epochs):
            np.random.shuffle(dataset)
            costs = []

            for s, sequence in enumerate(dataset):
                for i in xrange(0, len(sequence), batch_size):
                    cost = self.train_function(sequence[i:i + batch_size])
                    costs.append(cost)

            print 'Epoch %i/%i. Costs=%s' % (epoch + 1, num_epochs, np.mean(costs))
        training_minutes = (time.time() - start_time)/60.
        print("Training took %.1fmin. %.1fmin per epoch." % (training_minutes, training_minutes/num_epochs))

    def generate(self, filename, show=True):
        '''Generate a sample sequence, plot the resulting piano-roll and save
        it as a MIDI file.

        filename : string
            A MIDI file will be created at this location.
        show : boolean
            If True, a piano-roll of the generated sequence will be shown.'''

        piano_roll = self.generate_function()
        midiwrite(filename, piano_roll, self._pitch_range, self._midi_sampling_period)
        if show:
            extent = (0, self._midi_sampling_period * len(piano_roll)) + self._pitch_range
            plots.figure()
            plots.imshow(piano_roll.T, origin='lower', aspect='auto',
                         interpolation='nearest', cmap=plots.cm.gray_r,
                         extent=extent)
            plots.xlabel('time (s)')
            plots.ylabel('MIDI note number')
            plots.title('generated piano-roll')

    def get_params(self):
        return self._params


def run_rnn_rbm_training(trained_model_filename, reuse_pretrained=False, num_epochs = 1):
    batch_size = 100
    train_set_files, valid_set_files, test_set_files = get_dataset('nottingham')

    if reuse_pretrained:
        with open(trained_model_filename, 'r') as f:
            trained_model_params = cPickle.load(f)
    else:
        trained_model_params = None

    model = RnnRBM(network_parameters=trained_model_params)
    model.train(train_set_files, batch_size, num_epochs)

    with open(trained_model_filename, 'w') as f:
        cPickle.dump(model.get_params(), f, cPickle.HIGHEST_PROTOCOL)


def run_rnn_rbm_generation(trained_model_filename):
    with open(trained_model_filename, 'r') as f:
        trained_model_params = cPickle.load(f)

    trained_model = RnnRBM(network_parameters=trained_model_params)
    sample_filename = os.path.join(os.path.dirname(__file__), 'generated_sample1.mid')
    trained_model.generate(sample_filename)
    plots.show()
    play_midi(sample_filename)


if __name__ == "__main__":
    trained_model_filename = 'trained_rnn_rbm_params_100_500hidden.pkl'
    #run_rnn_rbm_training(trained_model_filename, reuse_pretrained=False, num_epochs=100)
    run_rnn_rbm_generation(trained_model_filename)
