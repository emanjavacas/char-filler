# coding: utf-8

import numpy as np
from keras.utils.np_utils import to_categorical
from models.lstms import simple_lstm, bilstm
from corpus import Indexer, Corpus

if __name__ == '__main__':
    idxr = Indexer(reserved={0: 'padding', 1: 'OOV'})
    train = Corpus('/Users/quique/corpora/EEBO_sample/train')
    test = Corpus('/Users/quique/corpora/EEBO_sample/test')
    dev = Corpus('/Users/quique/corpora/EEBO_sample/dev')

    print("Building encoder on train corpus")
    idxr.encode_seq(train.chars())  # quick pass to fit vocab

    print("Encoding test set")
    X_test, y_test = list(zip(*test.generate(idxr, ignore_oovs=True, oov_idx=1)))
    X_test = np.asarray(X_test),
    y_test = to_categorical(y_test, nb_classes=idxr.vocab_len())

    print("Encoding dev set")
    X_dev, y_dev = list(zip(*dev.generate(idxr, ignore_oovs=True, oov_idx=1)))
    X_dev = np.asarray(X_dev),
    y_dev = to_categorical(y_dev, nb_classes=idxr.vocab_len())

    print("Compiling model")
    model = simple_lstm(idxr.vocab_len())
    epochs = 1000

    for e in range(epochs):
        for X, y in train.generate_batches(idxr, ignore_oovs=True, oov_idx=1):
            X = np.asarray(X, dtype='float32')
            y = to_categorical(y, nb_classes=idxr.vocab_len())
            loss = model.train_on_batch(X, y)
