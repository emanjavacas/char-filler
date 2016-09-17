# coding: utf-8

import os
import random
from argparse import ArgumentParser

from sklearn.metrics import accuracy_score

from corpus import Indexer, Corpus
from unsmoothed_lm import UnsmoothedLM

from canister.experiment import Experiment


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-d', '--db', type=str, default='db.json')
    parser.add_argument('-n', '--num_lines', type=int, default=10000)
    parser.add_argument('-o', '--order', type=int, default=6)

    args = parser.parse_args()
    root = args.root
    path = args.db
    assert os.path.isdir(root), "Root path doesn't exist"

    ORDER = args.order
    NUM_LINES = args.num_lines

    idxr = Indexer(reserved={0: 'padding', 1: 'OOV'})
    train = Corpus(os.path.join(root, 'train'), side='left', context=ORDER)
    test = Corpus(os.path.join(root, 'test'), side='left', context=ORDER)

    print("Building encoder on train corpus")
    corpus = list(train.chars())
    idxr.encode_seq(corpus)  # quick pass to fit vocab
    n_chars = idxr.vocab_len()
    del corpus

    print("Training language model")
    model = UnsmoothedLM(order=ORDER)
    model.train(train.generate(idxr, oov_idx=1))

    tags = ('lm', 'seq')
    params = {
        'order': ORDER,
        'num_lines': NUM_LINES
    }
    db = Experiment.use(path, tags=tags, exp_id="char-fill").model('lm')
    y_random_true, y_random, y_ignore_true, y_ignore = [], [], [], []
    for hist, target in test.generate(idxr, oov_idx=1):
        # random guess if missing pred
        if random() > 0.005:  # downsample test corpus (1748471 examples)
            continue
        y_random.append(model.predict(tuple(hist), ensure_pred=True))
        y_random_true.append(target)
        # ignore target if missing pred
        try:
            y_ignore.append(model.predict(tuple(hist), ensure_pred=False))
            y_ignore_true.append(target)
        except KeyError:  # ignore
            pass

    db.add_result({
        'y_random_true': [idxr.decode(x) for x in y_random_true],
        'y_random': [idxr.decode(x) for x in y_random],
        'y_ignore_true': [idxr.decode(x) for x in y_ignore_true],
        'y_ignore': [idxr.decode(x) for x in y_ignore],
        'random_acc': accuracy_score(y_random_true, y_random),
        'ignore_acc': accuracy_score(y_ignore_true, y_ignore)
    }, params=params)

    print("Accuracy on ignore [%f]" % accuracy_score(y_ignore_true, y_ignore))
    print("Accuracy on random [%f]" % accuracy_score(y_random_true, y_random))
