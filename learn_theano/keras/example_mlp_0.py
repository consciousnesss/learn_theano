
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils


def run_mlp(n_epochs = 20):
    np.random.seed(1234)

    nb_inputs = 28*28
    nb_classes = 10

    model = Sequential()
    model.add(Dense(512, input_shape=(nb_inputs,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes, activation='softmax'))

    # RMSProp works way faster than SGD
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop())

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 28*28).astype("float32")/255
    X_test = X_test.reshape(10000, 28*28).astype("float32")/255
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    model.fit(X_train, Y_train, nb_epoch=n_epochs, batch_size=128,
              show_accuracy=True, verbose=2,
              validation_split=0.1)
    loss, accuracy = model.evaluate(X_test, Y_test,
                                    show_accuracy=True, verbose=0)
    return accuracy


if __name__ == '__main__':
    accuracy = run_mlp()
    # Expected: Test accuracy=98.590%
    print("Test accuracy=%.3f%%" % (accuracy*100,))
