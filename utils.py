#!/usr/bin/env python

import json
import os


def log_batch(epoch, batch, avg_train, train_loss, dev_loss, dev_acc):
    msg = "Epoch [%d], batch [%d], avg.tr. loss [%.3f], " + \
          "Tr. loss [%.3f], dev loss [%.3f], dev acc [%.3f]"
    print(msg % (epoch, batch, avg_train, train_loss, dev_loss, dev_acc),
          end='\r')


def one_hot(m, n_classes):
    "transforms a matrix into a one-hot encoded binary 3D tensor"
    if isinstance(m, (list, tuple)):
        m = np.asarray(m)
    return (np.arange(n_classes) == m[:, :, None]-1).astype(int)


def lines_from_file(f):
    if os.path.isfile(f):
        for line in open(f).readlines():
            yield line
    elif os.path.isdir(f):
        for ff in os.listdir(f):
            for line in lines_from_file(os.path.join(f, ff)):
                yield line
    else:  # assume is text
        for line in f.split('\n'):
            yield line + '\n'


if __name__ == '__main__':
    from argparse import ArgumentParser
    import numpy as np

    from corpus import Corpus, Indexer
    from keras.utils.np_utils import to_categorical

    parser = ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=True)
    args = parser.parse_args()

    root = args.root

    idxr = Indexer(reserved={0: 'padding', 1: 'OOV'})
    train = Corpus(root + 'train')
    test = Corpus(root + 'test')
    dev = Corpus(root + 'dev')

    idxr.encode_seq(train.chars())  # quick pass to fit vocab

    print("Encoding test set")
    X_test, y_test = list(zip(*test.generate(idxr, oov_idx=1)))
    X_test = np.asarray(X_test),
    y_test = to_categorical(y_test, nb_classes=idxr.vocab_len())

    print("Encoding dev set")
    X_dev, y_dev = list(zip(*dev.generate(idxr, oov_idx=1)))
    X_dev = np.asarray(X_dev),
    y_dev = to_categorical(y_dev, nb_classes=idxr.vocab_len())
