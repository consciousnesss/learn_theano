import numpy as np
import theano.tensor as T
import theano


def test_1_examples_sigmoid():
    x = T.dmatrix('x')
    s = 1/(1 + T.exp(-x))
    logistic = theano.function([x], s)
    np.testing.assert_array_almost_equal(
        logistic([[1, 1], [0., 2]]),
        [[0.73105858, 0.73105858], [0.5, 0.880797]]
    )


def test_1_examples_compute_more_than_1_return_value():
    a, b = T.dmatrices(['a', 'b'])
    diff = a - b
    abs_diff = abs(diff)
    diff_squared = diff**2
    f = theano.function([a, b], [diff, abs_diff, diff_squared])

    diff_res, abs_res, diff_squared_res = f([[1, 1], [1, 1]], [[0, 0], [2, 2]])
    np.testing.assert_array_almost_equal(diff_res, [[1, 1], [-1, -1]])
    np.testing.assert_array_almost_equal(abs_res, [[1, 1], [1, 1]])
    np.testing.assert_array_almost_equal(diff_squared_res, [[1, 1], [1, 1]])


if __name__ == "__main__":
    test_1_examples_compute_more_than_1_return_value()
