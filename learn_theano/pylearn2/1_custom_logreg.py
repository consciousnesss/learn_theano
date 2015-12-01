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


class MyLogisticRegression(Model):
    def __init__(self, n_vis_units, n_classes):
        Model.__init__(self)

        self._n_vis_units = n_vis_units
        self._n_classes = n_classes

        self._W = sharedX(np.random.uniform(size=(n_vis_units, n_classes)), 'W')
        self._b = sharedX(np.zeros(n_classes), 'b')

        # base class overrides
        self.input_space = VectorSpace(dim=n_vis_units)
        self.output_space = VectorSpace(dim=n_classes)

    def custom_logistic_regression(self, inputs):
        return T.nnet.softmax(T.dot(inputs, self._W) + self._b)

    def get_params(self):
        return [self._W, self._b]

    def get_monitoring_data_specs(self):
        '''
        what does this class needs to be passed to get_monitoring_channels
        '''
        space = CompositeSpace([self.input_space, self.output_space])
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    def get_monitoring_channels(self, data):
        space, source = self.get_monitoring_data_specs()
        space.validate(data)

        X, y = data
        y_hat = self.custom_logistic_regression(X)
        error = T.neq(y.argmax(axis=1), y_hat.argmax(axis=1)).mean()

        return OrderedDict([('error', error)])


class MyLogisticRegressionCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data, ** kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data

        # notice how customized model interface is
        outputs = model.custom_logistic_regression(inputs)
        loss = -(targets * T.log(outputs)).sum(axis=1)
        return loss.mean()


if __name__ == '__main__':
    dataset = MNIST(which_set='train', start=0, stop=50000)

    experiment = pylearn2.train.Train(
        dataset=dataset,
        model=MyLogisticRegression(n_vis_units=784, n_classes=10),
        algorithm=pylearn2.training_algorithms.sgd.SGD(
            batch_size=200,
            learning_rate=0.001,
            monitoring_dataset={
                'train': dataset,
                'valid': MNIST(which_set='train', start=50000, stop=60000),
                'test': MNIST(which_set='test')},
            cost=MyLogisticRegressionCost(),
            termination_criterion=pylearn2.termination_criteria.EpochCounter(max_epochs=50)
        ),
    )
    experiment.main_loop()
