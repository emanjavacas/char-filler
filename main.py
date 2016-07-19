# coding: utf-8

import numpy as np
from keras.utils.np_utils import to_categorical
from argparse import ArgumentParser


from models.lstms import simple_lstm, bilstm
from corpus import Indexer, Corpus

from utils import root, take

# Params
BATCH_SIZE = 128
NUM_BATCHES = 10000


def build_set(corpus, idxr, size=2000):
    dataset = take(corpus.generate(idxr, oov_idx=1), size)
    X, y = list(zip(*dataset))
    X = np.asarray(X)
    X = X.reshape(X.shape + (1,))
    y = to_categorical(y, nb_classes=idxr.vocab_len())
    return X, y


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--root', type=str)
    args = parser.parse_args()
    root = args.root if args.root else root

    idxr = Indexer(reserved={0: 'padding', 1: 'OOV'})
    train = Corpus(root + 'train')
    test = Corpus(root + 'test')
    dev = Corpus(root + 'dev')

    print("Building encoder on train corpus")
    corpus = list(train.chars())
    idxr.encode_seq(corpus)  # quick pass to fit vocab
    del corpus

    print("Encoding test set")
    X_test, y_test = build_set(test, idxr)

    print("Encoding dev set")
    X_dev, y_dev = build_set(dev, idxr)

    print("Compiling model")
    model = bilstm(idxr.vocab_len(), metrics=['accuracy'])

    print("Starting training")
    epochs = 5

    for e in range(epochs):
        losses = []
        batches = take(
            train.generate_batches(idxr, batch_size=BATCH_SIZE, oov_idx=1),
            NUM_BATCHES)
        for b, (X, y) in enumerate(batches):
            X = np.asarray(X, dtype='float32').reshape((BATCH_SIZE, -1, 1))
            y = to_categorical(y, nb_classes=idxr.vocab_len())
            loss, _ = model.train_on_batch(X, y)
            losses.append(loss)
            if b % 50 == 0:
                _, dev_acc = model.test_on_batch(X_dev, y_dev)
                print("Epoch: %d, Loss: %.4f, Dev acc: %.4f" %
                      (e, np.mean(losses), dev_acc))
    # Final
    _, test_acc = model.test_on_batch(X_test, y_test)
    print("Test acc: %.4f" % test_acc)
    model.save_weights('bilstm.h5')
