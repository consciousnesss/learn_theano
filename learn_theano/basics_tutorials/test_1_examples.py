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
    a, b = T.dmatrices('a', 'b')
    diff = a - b
    abs_diff = abs(diff)
    diff_squared = diff**2
    f = theano.function([a, b], [diff, abs_diff, diff_squared])

    diff_res, abs_res, diff_squared_res = f([[1, 1], [1, 1]], [[0, 0], [2, 2]])
    np.testing.assert_array_almost_equal(diff_res, [[1, 1], [-1, -1]])
    np.testing.assert_array_almost_equal(abs_res, [[1, 1], [1, 1]])
    np.testing.assert_array_almost_equal(diff_squared_res, [[1, 1], [1, 1]])


def test_1_examples_param_default():
    x, y = T.dscalars('x', 'y')
    f = theano.function([x, theano.Param(y, default=1)], x + y)
    assert f(1, 2) == 3
    assert f(1) == 2


def test_1_examples_accumulator():
    state = theano.shared(0)
    inc = T.iscalar('inc')
    accumulator = theano.function([inc], state, updates=[(state, state + inc)])
    assert state.get_value() == 0
    accumulator(1)
    assert state.get_value() == 1
    accumulator(100)
    assert state.get_value() == 101

    state.set_value(-1)
    assert state.get_value() == -1
    accumulator(2)
    assert state.get_value() == 1

    dec = T.iscalar('dec')
    decrementor = theano.function([dec], state, updates=[(state, state - dec)])
    decrementor(100)
    assert state.get_value() == -99


def test_1_examples_random_streams():
    random_stream = T.shared_randomstreams.RandomStreams(seed=243)
    uniform = random_stream.uniform((2, 2))
    normal = random_stream.normal((2, 2))
    # updates random gen state
    f = theano.function([], uniform)
    # doesn't update random gen state
    g = theano.function([], normal, no_default_updates=True)

    assert (f() != f()).all()
    assert (g() == g()).all()


def test_1_logistic_regression():
    N = 400
    features = 784
    training_steps = 2000
    inputs = np.random.randn(N, features).astype(theano.config.floatX)
    labels = np.random.randint(size=N, low=0, high=2).astype(np.int32)

    x = T.matrix('x')
    y = T.vector('y')
    w = theano.shared(np.random.randn(features), name='w')
    b = theano.shared(0., name='b')

    probability = 1/(1 + T.exp(-T.dot(x, w) + b))
    prediction = probability > 0.5

    cross_entropy = -y*T.log(probability) - (1-y)*T.log(1 - probability)
    cost = cross_entropy.mean() + 0.01*(w**2).sum()
    gradient_w, gradient_b = T.grad(cost, [w, b])

    train = theano.function([x, y], [prediction, cross_entropy],
                            updates=[(w, w - 0.1*gradient_w), (b, b-0.1*gradient_b)])
    predict = theano.function([x], prediction)
    pred, err = train(inputs, labels)
    assert(err.mean() > 5)
    for i in range(training_steps):
        pred, err = train(inputs, labels)

    assert(err.mean() < 0.07)
    assert((labels == predict(inputs)).all())


if __name__ == "__main__":
    test_1_logistic_regression()
