import numpy as np
import theano.tensor as T
import theano


def test_3_configuring():
    '''
    Example to configure logic regression to work with float32
    '''

    theano.config.floatX = 'float32'

    N = 400
    features = 784
    training_steps = 2000
    inputs = np.random.randn(N, features).astype(theano.config.floatX)
    labels = np.random.randint(size=N, low=0, high=2).astype(theano.config.floatX)

    x = T.matrix('x')
    y = T.vector('y')
    w = theano.shared(np.random.randn(features).astype(theano.config.floatX), name='w')
    b = theano.shared(np.asarray(0.).astype(theano.config.floatX), name='b')

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

    # figure out what did theano use to run this
    if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
            train.maker.fgraph.toposort()]):
        used = "cpu"
    elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
              train.maker.fgraph.toposort()]):
        used = "gpu"
    else:
        raise Exception('ERROR, not able to tell if theano used the cpu or the gpu')

    # can not use gpu on travis
    assert(used == "cpu")
    assert(theano.config.floatX == 'float32')


if __name__ == "__main__":
    test_3_configuring()
