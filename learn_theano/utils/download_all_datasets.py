from learn_theano.utils.s3_download import S3


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


if __name__ == "__main__":
    download_all_datasets()
