# coding: utf-8

from collections import defaultdict, Counter
from random import random, randint

from corpus import Corpus, Indexer


class UnsmoothedLM(object):
    def __init__(self, order=6):
        self.order = order
        self.lm = defaultdict(Counter)
        self.random = {}

    def train(self, corpus, **kwargs):
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
            normalize(counts)

    def _random_dist(self):
        assert self.random, "Model hasn't been trained yet"
        random_prefix = self.random[randint(0, len(self.lm) - 1)]
        assert random_prefix in self.lm
        return self.lm[random_prefix]

    def generate_char(self, hist, ensure_char=True):
        if tuple(hist) not in self.lm:
            if ensure_char:  # randomly sample a distribution.
                dist = self._random_dist()
            else:
                raise KeyError("couldn't find hist %s" % str(hist))
        else:
            dist = self.lm[tuple(hist)]
        x = random()  # simple sampling from distribution
        for char, v in dist.items():
            x -= v
            if x <= 0:
                return char

    def predict(self, prefix, ensure_pred=True):
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

    def generate_text(self, nletters=1000, idxr=None, **kwargs):
        text = []
        hist = [0] * self.order  # start with a seq of padding
        for i in range(nletters):
            c = self.generate_char(hist, **kwargs)
            hist = hist[-self.order + 1:] + [c]
            text.append(c)
        if idxr is not None:
            return "".join([idxr.decode(x) for x in text])
        else:
            return text


if __name__ == '__main__':
    shakespeare = 'http://cs.stanford.edu/people/karpathy/' + \
                  'char-rnn/shakespeare_input.txt'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--order', default=6, type=int)
    parser.add_argument('-u', '--url', nargs='+')
    parser.add_argument('-f', '--file', nargs='+')

    args = parser.parse_args()

    from six.moves.urllib import request
    import sys

    def read_urls(*urls):
        text = []
        for url in urls:
            print("Downloading [%s]" % url)
            try:
                req = request.Request(url)
                with request.urlopen(req) as f:
                    for line in f.read().decode('utf-8').split('\r\n'):
                        text += [line + "\n"]
            except ValueError:
                print("Couldn't download [%s]" % url)
        return text

    def read_files(*files):
        import os
        text = []
        for fl in files:
            if os.path.isfile(fl):
                with open(fl, mode='r', encoding='utf-8') as f:
                    print("Reading [%s]" % fl)
                    for line in f.read().split('\n'):
                        text += [line + '\n']
            elif os.path.isdir(fl):
                for ffl in os.listdir(fl):
                    flpath = os.path.join(fl, ffl)
                    if not os.path.isfile(flpath):
                        print("Ignoring [%s]" % flpath)
                        continue
                    with open(flpath, mode='r', encoding='utf-8') as f:
                        print("Reading [%s]" % flpath)
                        for line in f.read().split('\n'):
                            text += [line + '\n']
        return text

    print("Fetching texts")
    text = []
    if args.url:
        text += read_urls(*args.url)
    if args.file:
        text += read_files(*args.file)
    if not text:
        print("No input text, exiting...")
        sys.exit(0)

    lines = (line for line in text if line.strip())
    corpus = Corpus(lines, context=args.order, side='left')
    idxr, model = Indexer(), UnsmoothedLM(order=args.order)

    print("Training on corpus")
    model.train(corpus.generate(idxr))
    del text

    def ensure_res(question, validators, msg, prompt='>>> '):
        print(question)
        res = input(prompt)
        while not any(filter(lambda x: x(res), validators)):
            print(msg)
            res = input(prompt)
        return res

    question = 'generate text (y) or quit (n)?\n'
    msg = 'Sorry, input must be ("y", "n")'
    validators = [lambda x: x in ('y', 'n')]

    res = None
    while res != 'n':
        print("------ Generating text -------")
        print("-----------------------------")
        print(model.generate_text(idxr=idxr, ensure_char=True) + "\n")
        print("--- End of Generated text ---")
        print("-----------------------------")
        res = ensure_res(question, validators, msg)

    print("bye!")
    sys.exit(0)
