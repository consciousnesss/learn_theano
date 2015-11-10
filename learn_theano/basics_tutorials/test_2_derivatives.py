import theano.tensor as T


def test_2_gradient():
    x = T.dscalar('x')
    x_sq = x*x
    gradient, = T.grad(x_sq, [x])

    assert gradient.eval({x: 2}) == 4


if __name__ == '__main__':
    test_2_gradient()
