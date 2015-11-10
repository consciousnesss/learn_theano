from learn_theano.utils.download_all_datasets import get_dataset
import os


def test_check_mnist():
    mnist_pkl = get_dataset('mnist')
    assert(os.path.isfile(mnist_pkl))
