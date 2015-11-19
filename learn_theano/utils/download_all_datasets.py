from learn_theano.utils.midi_nottingham_dataset import get_nottingham_midi_folder
from learn_theano.utils.s3_download import S3
import os
import cv2
import cPickle


datasets = {
    'mnist': lambda: get_standard_pickled_dataset('datasets/mnist.pkl.gz'),
    'imdb': lambda: get_standard_pickled_dataset('datasets/imdb_filtered.pkl.gz'),
    'nottingham': get_nottingham_midi_folder
}


def download_all_datasets():
    for d in datasets:
        get_dataset(d)


def get_standard_pickled_dataset(artifact_name):
    filename = S3().download(artifact_name)
    with open(filename) as f:
        return cPickle.load(f)


def get_dataset(name):
    return datasets[name]()


def get_3_wolves_image():
    filename = os.path.join(os.path.dirname(__file__), 'artifacts/3wolfmoon.jpg')
    return cv2.imread(filename)


if __name__ == "__main__":
    download_all_datasets()
