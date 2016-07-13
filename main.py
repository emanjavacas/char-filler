# coding: utf-8

import numpy as np
from keras.utils.np_utils import to_categorical
from models.lstm import simple_lstm, bilstm
from corpus import Indexer, Corpus

if __name__ == '__main__':
    idxr = Indexer(reserved={0: 'padding', 1: 'OOV'})
    train = Corpus('/Users/quique/corpora/EEBO_sample/train')
    test = Corpus('/Users/quique/corpora/EEBO_sample/test')

    idxr.encode_seq(train.chars())  # quick pass to fit vocab

    X_test, y_test = list(zip(test.generate()))
    X_test = np.asarray(X_test),
    y_test = to_categorical(y_test, nb_classes=idxr.vocab_len())

    model = simple_lstm(idxr.vocab_len())
    epochs = 1000

    for e in epochs:
        for X, y in train.generate_batches(idxr, ignore_oovs=True, oov_idx=1):
            X = np.asarray(X)
            y = np.to_categorical(y, nb_classes=idxr.vocab_len())
