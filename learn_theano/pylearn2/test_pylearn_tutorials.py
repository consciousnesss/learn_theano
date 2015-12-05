from learn_theano.pylearn2.custom_autoencoder_2 import custom_autoencoder_run
from learn_theano.pylearn2.custom_logreg_1 import custom_log_reg_run
from learn_theano.pylearn2.standard_mlp_0 import standard_mlp_run
import theano


def test_standard_mlp():
    standard_mlp_run(1)


def test_custom_log_reg():
    old_floatx = theano.config.floatX
    theano.config.floatX = 'float64'
    custom_log_reg_run(1)
    theano.config.floatX = old_floatx


def test_custom_autoencoder():
    custom_autoencoder_run(1)
