from learn_theano.utils.s3_download import S3
import os
import cv2


datasets = {
    'mnist': 'datasets/mnist.pkl.gz'
}


def download_all_datasets():
    filenames = []
    for d in datasets:
        filenames.append(get_dataset(d))
    print("Downloaded the following datasets: %s" % (filenames,))


def get_dataset(name):
    return S3().download(datasets[name])


def get_3_wolves_image():
    filename = os.path.join(os.path.dirname(__file__), 'artifacts/3wolfmoon.jpg')
    return cv2.imread(filename)


if __name__ == "__main__":
    download_all_datasets()
