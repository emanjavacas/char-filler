#!/usr/bin/env python

import json


def save_ndarray(ndarray, fname):
    with open(fname, 'wb') as f:
        ndarray.dump(f)


def dump_json(obj, fname):
    with open(fname, 'w') as f:
        json.dump(obj, f)


def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)


def take(g, n):
    index = 0
    for x in g:
        if index >= n:
            break
        yield x
        index += 1


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
