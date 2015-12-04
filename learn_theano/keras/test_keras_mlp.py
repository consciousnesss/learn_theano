from learn_theano.keras.example_mlp_0 import run_mlp


def test_keras_mlp():
    accuracy = run_mlp(n_epochs=1)
    assert(accuracy > 0.95)


if __name__ == '__main__':
    test_keras_mlp()
