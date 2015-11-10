from learn_theano.utils.download_all_datasets import get_dataset
import os
import cPickle
import numpy as np


def test_check_mnist():
    mnist_pkl = get_dataset('mnist')
    assert(os.path.isfile(mnist_pkl))
    with open(mnist_pkl) as f:
        train_set, valid_set, test_set = cPickle.load(f)
    assert (len(train_set), len(valid_set), len(test_set)) == (2, 2, 2)
    # mnist pictures are 28x28 = 784 of float grayscale values
    assert (train_set[0].shape, train_set[1].shape) == ((50000, 784), (50000,))
    assert (valid_set[0].shape, valid_set[1].shape) == ((10000, 784), (10000,))
    assert (test_set[0].shape, test_set[1].shape) == ((10000, 784), (10000,))
    assert (train_set[0].dtype, train_set[1].dtype) == (np.float32, np.int64)


if __name__ == "__main__":
    test_check_mnist()
