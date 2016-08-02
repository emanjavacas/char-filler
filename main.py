# coding: utf-8

import os
from inspect import getsourcefile
from argparse import ArgumentParser

import numpy as np
from keras.utils.np_utils import to_categorical

from models import lstms
from corpus import Indexer, Corpus
from utils import take, dump_json
from canister.experiment import Experiment


BATCH_MSG = "Epoch: %d, Loss: %.4f, Dev-loss: %.4f: Dev-acc: %.4f"


def build_set(corpus, idxr, size=2000):
    dataset = take(corpus.generate(idxr, oov_idx=1), size)
    X, y = list(zip(*dataset))
    X = np.asarray(X)
    X = X.reshape(X.shape + (1,))
    y = to_categorical(y, nb_classes=idxr.vocab_len())
    return X, y


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-n', '--num_batches', type=int, default=10000)
    parser.add_argument('-p', '--model_prefix', type=str, required=True)
    parser.add_argument('-d', '--db', type=str, default='db.json')

    args = parser.parse_args()
    root = args.root
    assert os.path.isdir(root), "Root path doesn't exist"

    BATCH_SIZE = args.batch_size
    NUM_BATCHES = args.num_batches
    EPOCHS = args.epochs

    idxr = Indexer(reserved={0: 'padding', 1: 'OOV'})
    train = Corpus(os.path.join(root, 'train'))
    test = Corpus(os.path.join(root, 'test'))
    dev = Corpus(os.path.join(root, 'dev'))

    print("Building encoder on train corpus")
    corpus = list(train.chars())
    idxr.encode_seq(corpus)  # quick pass to fit vocab
    del corpus

    print("Encoding test set")
    X_test, y_test = build_set(test, idxr)

    print("Encoding dev set")
    X_dev, y_dev = build_set(dev, idxr)

    print("Compiling model")
    model = lstms.bilstm(idxr.vocab_len(), metrics=['accuracy'])

    # experiment dbxs
    tags = ('lstm', 'seq')
    exp_id = getsourcefile(lambda: 0)
    params = {'batch_size': BATCH_SIZE, 'num_batches': NUM_BATCHES}
    model_db = Experiment.use(args.db, root, tags=tags, exp_id=exp_id) \
                         .model("bilstm", model.get_config())

    print("Starting training")
    with model_db.session(params) as session:
        from time import time
        start = time()
        for e in range(EPOCHS):
            losses = []
            batches = take(
                train.generate_batches(idxr, batch_size=BATCH_SIZE, oov_idx=1),
                NUM_BATCHES)
            for b, (X, y) in enumerate(batches):
                X = np.asarray(X, dtype='float32').reshape((BATCH_SIZE, -1, 1))
                y = to_categorical(y, nb_classes=idxr.vocab_len())
                loss, _ = model.train_on_batch(X, y)
                losses.append(loss)
                if b % 150 == 0:
                    dev_loss, dev_acc = model.test_on_batch(X_dev, y_dev)
                    print(BATCH_MSG % (e, np.mean(losses), dev_loss, dev_acc))
            session.add_epoch(e, {'training_loss': np.mean(losses),
                                  'dev_loss': dev_loss,
                                  'dev_acc': dev_acc})
        _, test_acc = model.test_on_batch(X_test, y_test)
        print("Test acc: %.4f" % test_acc)
        session.add_result({'test_acc': test_acc})
        session.add_meta({'run_time': time() - start,
                          'model_prefix': args.model_prefix})

    # save model + indexer
    model.save_weights(args.model_prefix + '_weights.h5')
    dump_json(model.get_config(), args.model_prefix + '_config.json')
    idxr.save(args.model_prefix + '_indexer.json')
