# coding: utf-8

import os
from random import random
from argparse import ArgumentParser

from sklearn.metrics import accuracy_score
from utils import lines_from_file
from unsmoothed_lm import UnsmoothedLM, generate_pairs
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

    train = generate_pairs(lines_from_file(os.path.join(root, 'train')), order=ORDER)
    test = generate_pairs(lines_from_file(os.path.join(root, 'test')), order=ORDER)

    print("Training language model")
    model = UnsmoothedLM(order=ORDER)
    model.train(train)

    tags = ('lm', 'seq')
    params = {
        'order': ORDER,
        'num_lines': NUM_LINES
    }
    db = Experiment.use(path, tags=tags, exp_id="char-fill").model('lm')
    y_random_true, y_random, y_ignore_true, y_ignore = [], [], [], []
    for hist, target in test:
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
        'y_random_true': y_random_true,
        'y_random': y_random,
        'y_ignore_true': y_ignore_true,
        'y_ignore': y_ignore,
        'random_acc': accuracy_score(y_random_true, y_random),
        'ignore_acc': accuracy_score(y_ignore_true, y_ignore)
    }, params=params)

    print("Accuracy on ignore [%f]" % accuracy_score(y_ignore_true, y_ignore))
    print("Accuracy on random [%f]" % accuracy_score(y_random_true, y_random))
