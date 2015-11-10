from learn_theano.utils.download_all_datasets import download_all_datasets


def pytest_configure(config):
    download_all_datasets()
