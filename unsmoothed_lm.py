# coding: utf-8

import os
from random import random
from argparse import ArgumentParser

from sklearn.metrics import accuracy_score
from utils import lines_from_file
from unsmoothed_lm import UnsmoothedLM, generate_pairs
from casket.experiment import Experiment

from collections import defaultdict, Counter
from random import random, randint


def generate_pairs(lines, order, pad="~"):
    padding = pad * order
    for line in lines:
        line = line.strip().replace(pad, "-")
        if not line:
            continue
        line = padding + line + padding
        for i in range(len(line) - order):
            yield line[i: i+order], line[i+order]


class UnsmoothedLM(object):
    def __init__(self, order=6):
        self.order = order
        self.lm = defaultdict(Counter)
        self.random = {}

    def train(self, corpus):
        for hist, target in corpus:
            self.lm[tuple(hist)][target] += 1

        def normalize(counts):
            total = sum(counts.values())
            for target, cnt in counts.items():
                counts[target] = cnt/total

        from random import sample
        idxs = sample(range(len(self.lm)), len(self.lm))
        for hist, counts in self.lm.items():
            self.random[idxs.pop()] = hist  # add random index to lm
            normalize(counts)  # inplace modification

    def _random_dist(self):
        assert self.random, "Model hasn't been trained yet"
        random_prefix = self.random[randint(0, len(self.lm) - 1)]
        return self.lm[random_prefix]

    def generate_char(self, hist, ensure_char=False):
        if tuple(hist) not in self.lm:
            if ensure_char:  # randomly sample a distribution.
                dist = self._random_dist()
            else:
                raise KeyError("couldn't find hist %s" % str(hist))
        else:
            dist = self.lm[tuple(hist)]
        x = random()  # simple sampling from distribution
        for char, prob in dist.items():
            x -= prob
            if x <= 0:
                return char

    def predict(self, prefix, ensure_pred=False):
        assert len(prefix) == self.order, \
            "prefix must be of lm order [%d]" % self.order
        if prefix not in self.lm:
            if ensure_pred:
                dist = self._random_dist()
            else:
                raise KeyError(str(prefix))
        else:
            dist = self.lm[prefix]
        return dist.most_common(1)[0][0]  # return argmax

    def generate_text(self, nletters=1000, idxr=None, pad="~", **kwargs):
        assert pad or (idxr and idxr.pad)
        text = []
        hist = [pad or idxr.pad] * self.order  # start with a seq of padding
        for i in range(nletters):
            c = self.generate_char(hist, **kwargs)
            hist = hist[-self.order + 1:] + [c]
            text.append(c)
        if idxr is not None:
            out = "".join([idxr.decode(x) for x in text])
        else:
            out = "".join(text)
        return out.replace((pad or idxr.pad) * self.order, "\n")


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
