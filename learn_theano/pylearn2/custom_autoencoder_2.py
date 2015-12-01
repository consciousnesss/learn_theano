from collections import OrderedDict

import numpy as np
import theano.tensor as T
from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils import sharedX
import pylearn2.train
import pylearn2.training_algorithms.sgd
import pylearn2.termination_criteria


class MyAutoencoder(Model):
    def __init__(self, n_vis_units, n_hidden_units):
        Model.__init__(self)

        self._W = sharedX(np.random.uniform(size=(n_vis_units, n_hidden_units)), 'W')
        self._b = sharedX(np.zeros(n_hidden_units), 'b')
        self._b_reconstruction = sharedX(np.zeros(n_vis_units), 'b_reconstruction')
        self.input_space = VectorSpace(dim=n_vis_units)

    def my_reconstruct(self, X):
        h = T.tanh(T.dot(X, self._W) + self._b)
        return T.nnet.sigmoid(T.dot(h, self._W.T) + self._b_reconstruction)

    def get_params(self):
        return [self._W, self._b, self._b_reconstruction]


class MyAutoencoderCost(DefaultDataSpecsMixin, Cost):
    supervised = False  # cost will not receive labels

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        X = data
        X_hat = model.my_reconstruct(X)
        loss = -(X * T.log(X_hat) + (1 - X) * T.log(1 - X_hat)).sum(axis=1)
        return loss.mean()


def custom_autoencoder_run(max_epochs=50):
    dataset = MNIST(which_set='train', start=0, stop=50000)

    experiment = pylearn2.train.Train(
        dataset=dataset,
        model=MyAutoencoder(n_vis_units=784, n_hidden_units=200),
        algorithm=pylearn2.training_algorithms.sgd.SGD(
            batch_size=200,
            learning_rate=0.001,
            monitoring_dataset={
                'train': dataset,
                'valid': MNIST(which_set='train', start=50000, stop=60000),
                'test': MNIST(which_set='test')},
            cost=MyAutoencoderCost(),
            termination_criterion=pylearn2.termination_criteria.EpochCounter(max_epochs=50)
        ),
    )
    experiment.main_loop()


if __name__ == '__main__':
    custom_autoencoder_run()
