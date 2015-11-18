from learn_theano.utils.download_all_datasets import get_dataset


def run_7_lstm_training():
    train_set, valid_set, test_set = get_dataset('imdb')
    print(len(train_set[0]), len(valid_set), len(test_set))



if __name__ == '__main__':
    run_7_lstm_training()
