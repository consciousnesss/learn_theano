import theano.tensor as T
import theano
from theano.ifelse import ifelse
import numpy as np


def test_4_conditionals():
    a, b = T.scalars('a', 'b')
    x, y = T.matrices('x', 'y')

    f_switch = theano.function([a, b, x, y], T.switch(T.lt(a, b), T.mean(x), T.mean(y)))
    f_lazy_ifelse = theano.function([a, b, x, y], ifelse(T.lt(a, b), T.mean(x), T.mean(y)))

    x_val = np.ones((100, 100))*1
    y_val = np.ones((100, 100))*2

    # vectorized switch is going to evaluate both options
    np.testing.assert_almost_equal(
        f_switch(1, 2, x_val, y_val),
        1
    )

    # lazy evaluation is going to evaluate only single option
    np.testing.assert_almost_equal(
        f_lazy_ifelse(2, 1, x_val, y_val),
        2
    )


if __name__ == '__main__':
    test_4_conditionals()
