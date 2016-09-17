# coding: utf-8

import os
from argparse import ArgumentParser

import numpy as np
from keras.utils.np_utils import to_categorical

from corpus import Indexer, Corpus
from utils import take, dump_json
import lstms

from canister.experiment import Experiment


BATCH_MSG = "Epoch: %d, Loss: %.4f, Dev-loss: %.4f: Dev-acc: %.4f"


def one_hot(m, n_classes):
    "transformas a matrix into a one-hot encoded binary 3D tensor"
    if isinstance(m, (list, tuple)):
        m = np.asarray(m)
    return (np.arange(n_classes) == m[:, :, None]-1).astype(int)


def build_set(corpus, idxr, size=2000, one_hot_enc=True):
    dataset = take(corpus.generate(idxr, oov_idx=1), size)
    X, y = list(zip(*dataset))
    X = np.asarray(X)
    if one_hot_enc:
        X = one_hot(X, idxr.vocab_len())
    y = to_categorical(y, nb_classes=idxr.vocab_len())
    return X, y


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop')
    parser.add_argument('-R', '--rnn_layers', type=int, default=1)
    parser.add_argument('-m', '--emb_dim', type=int, default=28)
    parser.add_argument('-l', '--lstm_dim', type=int, default=100)
    parser.add_argument('-H', '--hidden_dim', type=int, default=250)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=50)
    parser.add_argument('-n', '--num_batches', type=int, default=10000)
    parser.add_argument('-p', '--model_prefix', type=str, required=True)
    parser.add_argument('-d', '--db', type=str, default='db.json')
    parser.add_argument('-L', '--loss', type=int, default=50,
                        help='report loss every l batches')

    args = parser.parse_args()
    root = args.root
    path = args.db
    assert os.path.isdir(root), "Root path doesn't exist"

    BATCH_SIZE = args.batch_size
    NUM_BATCHES = args.num_batches
    EPOCHS = args.epochs
    OPTIMIZER = args.optimizer
    RNN_LAYERS = args.rnn_layers
    EMB_DIM = args.emb_dim
    LSTM_DIM = args.lstm_dim
    HIDDEN_DIM = args.hidden_dim

    idxr = Indexer(reserved={0: 'padding', 1: 'OOV'})
    train = Corpus(os.path.join(root, 'train'))
    test = Corpus(os.path.join(root, 'test'))
    dev = Corpus(os.path.join(root, 'dev'))

    print("Building encoder on train corpus")
    corpus = list(train.chars())
    idxr.encode_seq(corpus)  # quick pass to fit vocab
    n_chars = idxr.vocab_len()
    del corpus

    print("Encoding test set")
    has_emb = args.model == "emb_bilstm"
    X_test, y_test = build_set(test, idxr, one_hot_enc=not has_emb)

    print("Encoding dev set")
    X_dev, y_dev = build_set(dev, idxr, one_hot_enc=not has_emb)

    print("Compiling model")
    params = {'rnn_layers': RNN_LAYERS, 'lstm_layer': LSTM_DIM,
              'hidden_layer': HIDDEN_DIM, 'optimizer': OPTIMIZER,
              'batch_size': BATCH_SIZE, 'num_batches': NUM_BATCHES}
    if args.model == 'bilstm':
        model = lstms.bilstm(n_chars,
                             rnn_layers=RNN_LAYERS, lstm_layer=LSTM_DIM,
                             hidden_layer=HIDDEN_DIM)
    elif args.model == 'emb_bilstm':
        params.update({'emb_dim': EMB_DIM})
        model = lstms.emb_bilstm(n_chars, EMB_DIM,
                                 rnn_layers=RNN_LAYERS, lstm_layer=LSTM_DIM,
                                 hidden_layer=HIDDEN_DIM)
    else:
        raise ValueError("Missing model [%s]" % args.model)

    model.compile(OPTIMIZER,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # experiment db
    tags = ('lstm', 'seq')
    db = Experiment.use(path, tags=tags, exp_id="char-fill").model(args.model)

    print("Starting training")
    with db.session(params, ensure_unique=False) as session:
        from time import time
        start = time()
        for e in range(EPOCHS):
            losses = []
            batches = take(
                train.generate_batches(idxr, batch_size=BATCH_SIZE, oov_idx=1),
                NUM_BATCHES)
            for b, (X, y) in enumerate(batches):
                X = np.asarray(X) if has_emb else one_hot(X, n_chars)
                y = to_categorical(y, nb_classes=n_chars)
                loss, _ = model.train_on_batch(X, y)
                losses.append(loss)
                if b % args.loss == 0:
                    dev_loss, dev_acc = model.test_on_batch(X_dev, y_dev)
                    print(BATCH_MSG % (e, np.mean(losses), dev_loss, dev_acc))
            session.add_epoch(
                e, {'training_loss': str(np.mean(losses)),
                    'dev_loss': str(dev_loss),
                    'dev_acc': str(dev_acc)})
        _, test_acc = model.test_on_batch(X_test, y_test)
        print("Test acc: %.4f" % test_acc)
        session.add_result({'test_acc': str(test_acc)})
        session.add_meta({'run_time': time() - start,
                          'model_prefix': args.model_prefix})

    # save model + indexer
    model.save_weights(args.model_prefix + '_weights.h5')
    dump_json(model.get_config(), args.model_prefix + '_config.json')
    idxr.save(args.model_prefix + '_indexer.json')