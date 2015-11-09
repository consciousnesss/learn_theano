from theano import function
import theano.tensor as T
import numpy as np


def test_0_baby_steps_scalars():
    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x + y
    f = function([x, y], z)
    assert f(2, 3) == 5


def test_0_baby_steps_matrices():
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x + y
    np.testing.assert_array_almost_equal(
        z.eval({x: np.array([[0., 1, 3], [1., 0, 3]]),
                y: np.array([[1., 1, 3], [1., 0, 3]])}),
        np.array([[1., 2, 6], [2., 0, 6]])
    )


if __name__ == '__main__':
    test_0_baby_steps_matrices()
