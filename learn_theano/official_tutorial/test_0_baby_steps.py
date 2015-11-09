from theano import function
import theano.tensor as T


def test_0_baby_steps():
    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x + y
    f = function([x, y], z)
    assert f(2, 3) == 5
