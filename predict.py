# coding: utf-8

import numpy as np

from casket.nlp_utils.corpus import pad


def get_max_n(arr, max_n=1):
    return arr.argsort()[-max_n:][::-1]


class CharFiller(object):
    def __init__(self, model, idxr, context):
        self.model = model
        self.idxr = idxr
        self.context = context

    def predict(self, s, pos, with_prob=False, max_n=1):
        pad_code = self.idxr.pad_code
        left = self.idxr.transform(s[max(0, pos - self.context): pos])
        left = pad(left, self.context, paditem=self.idxr.pad_code)
        right = self.idxr.transform(s[pos+1: min(len(s), pos+self.context)])
        right = pad(right, self.context, paditem=pad_code, paddir='right')
        hist = np.asarray(left + right).reshape((1, self.context * 2))
        pred = self.model.predict(hist)[0]  # model returns a embedded array
        best = get_max_n(pred, max_n=max_n)
        if with_prob:
            return [(self.idxr.decode(code), pred[code]) for code in best]
        else:
            return [self.idxr.decode(code) for code in best]


if __name__ == '__main__':
    import argparse
    from keras.models import load_model
    from lasagne_bilstm import BiRNN
    from casket.nlp_utils import Indexer

    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to model prefix')
    parser.add_argument('-t', '--type', required=True,
                        help='model type (keras/lasagne)')

    args = parser.parse_args()

    idxr = Indexer.load(args.model + '_indexer.json')

    if args.type == 'keras':
        model = load_model(args.model + '.h5')
    elif args.type == 'lasagne':
        model = BiRNN.load(args.model)
    else:
        raise ValueError("Didn't understand model [%s]" % args.model)

    filler = CharFiller(model, idxr, 10)

    sent = "this is a sentence to be filled with a lot of characters",
    filler.predict(sent, 15, max_n=5, with_prob=True)
