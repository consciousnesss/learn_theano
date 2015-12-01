from learn_theano.pylearn2.custom_autoencoder_2 import custom_autoencoder_run
from learn_theano.pylearn2.custom_logreg_1 import custom_log_reg_run
from learn_theano.pylearn2.standard_mlp_0 import standard_mlp_run


def test_standard_mlp():
    standard_mlp_run(2)


def test_custom_log_reg():
    custom_log_reg_run(2)


def test_custom_autoencoder():
    custom_autoencoder_run(2)
