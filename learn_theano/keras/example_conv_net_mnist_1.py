from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils


def run_mnist_conv_net():
    np.random.seed(1234)

    img_rows, img_cols = 28, 28
    nb_classes = 10

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='same',
                            input_shape=(1, img_rows, img_cols),
                            activation='relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    # RMSProp works way faster than SGD
    optimizer = 'adadelta'
    #optimizer = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train[:, None, :, :].astype("float32")/255
    X_test = X_test[:, None, :, :].astype("float32")/255
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    model.fit(X_train, Y_train, nb_epoch=12, batch_size=128,
              show_accuracy=True, verbose=2,
              validation_split=0.1)
    loss, accuracy = model.evaluate(X_test, Y_test,
                                    show_accuracy=True, verbose=0)
    return accuracy


if __name__ == '__main__':
    accuracy = run_mnist_conv_net()
    # Expected: Test accuracy=98.590%
    print("Test accuracy=%.3f%%" % (accuracy*100,))
