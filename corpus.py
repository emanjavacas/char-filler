# coding: utf-8

import os
import types
import pickle as p  # python3
import json
import logging


LOGGER = logging.getLogger(__name__)


def pad(items, maxlen, paditem=0, paddir='left'):
    """
    Parameters:
    -----------
    items: iterable, an iterable of objects to index
    maxlen: int, length to which input will be padded
    paditem: any, element to use for padding
    paddir: ('left', 'right'), where to add the padding
    """
    n_items = len(items)
    if n_items == maxlen:
        return items
    if paddir == 'left':
        return (maxlen - n_items) * [paditem] + items
    if paddir == 'right':
        return items + (maxlen - n_items) * [paditem]


def lines_from_file(fname):
    with open(fname, 'r') as f:
        for line in f:
            yield line


class Indexer(object):
    def __init__(self, pad='~', oov='±', verbose=False):
        """
        Parameters:
        -----------
        reserved: dict {int: any}, preindexed reserved items.
        This is useful in case of reserved values (e.g. padding).
        Indices must start at 0.

        Example:
        --------
        indexer = Indexer()
        """
        self.level = logging.WARN if verbose else logging.NOTSET
        self.pad = pad
        self.oov = oov
        self.decoder = {}
        self.encoder = {}
        self._current = 0
        if pad:
            self.encode(pad)
            self.pad = pad
        if oov:
            self.encode(oov)
            self.oov = oov

    def vocab(self):
        return self.encoder.keys()

    def vocab_len(self):
        return len(self.encoder)

    def set_verbose(self, verbose=True):
        self.level = logging.WARN if verbose else logging.NOTSET

    def encode(self, s):
        """
        Parameters:
        -----------
        s: object, object to index

        Returns:
        --------
        idx (int)
        """
        if s in self.encoder:
            return self.encoder[s]
        elif self.oov:
            return self.oov_encode
        else:
            LOGGER.log(self.level, "Inserting new item [%s]" % s)
            idx = self._current
            self.encoder[s] = idx
            self.decoder[idx] = s
            self._current += 1
            return idx

    def decode(self, idx):
        try:
            return self.decoder[idx]
        except KeyError:
            if self.oov:
                return self.oov_decode
            else:
                raise KeyError

    def encode_seq(self, seq, **kwargs):
        return [self.encode(x, **kwargs) for x in seq]

    def _to_json(self):
        obj = {'encoder': self.encoder, 'decoder': self.decoder}
        if self.pad:
            obj.update({'pad': self.pad})
        if self.oov:
            obj.update({'oov': self.oov})
        return obj

    def save(self, fname, mode='json'):
        if mode == 'json':
            with open(fname, 'w') as f:
                json.dump(self._to_json(), f)
        elif mode == 'pickle':
            with open(fname, 'wb') as f:
                p.dump(self, f)
        else:
            raise ValueError('Unrecognized mode %s' % mode)

    @staticmethod
    def load(fname, mode='json'):
        if mode == 'pickle':
            with open(fname, 'rb') as f:
                return p.load(f)
        elif mode == 'json':
            with open(fname, 'r') as f:
                return Indexer.from_dict(json.load(f))
        else:
            raise ValueError('Unrecognized mode %s' % mode)

    @classmethod
    def from_dict(cls, d, **kwargs):
        idxr = cls(**kwargs)
        idxr.encoder = d['encoder']
        idxr.decoder = {int(k): v for (k, v) in d['decoder']}
        idxr._current = len(idxr.encoder)
        if 'pad' in d:
            idxr.pad = d['pad']
        if 'oov' in d:
            idxr.oov = d['oov']
        return idxr


class Corpus(object):
    def __init__(self, root, context=10, side='both'):
        """
        Parameters:
        ------------
        root: str/generator, source of lines. If str, it is coerced to file/dir
        context: int, context characters around target char
        side: str, one of 'left', 'right', 'both', context origin
        """
        self.root = root
        self.context = context
        if side not in {'left', 'right', 'both'}:
            raise ValueError('Invalid side value [%s]' % side)
        self.side = side

    def _encode_line(self, line, indexer, **kwargs):
        encoded_line = [indexer.encode(c, **kwargs) for c in line]
        maxlen = len(encoded_line)
        for idx, c in enumerate(encoded_line):
            minidx = max(0, idx - self.context)
            maxidx = min(maxlen, idx + self.context + 1)
            if self.side in {'left', 'both'}:
                left = pad(encoded_line[minidx: idx], self.context,
                           paditem=indexer.pad, paddir='left')
            if self.side in {'right', 'both'}:
                right = pad(encoded_line[idx + 1: maxidx], self.context,
                            paditem=indexer.pad, paddir='right')
            if self.side == 'left':
                yield left, c
            elif self.side == 'right':
                yield right, c
            else:
                yield left + right, c

    def chars(self):
        """
        Returns:
        --------
        generator over characters in root
        """
        if isinstance(self.root, types.GeneratorType):
            for line in self.root:
                for char in line:
                    yield char
        elif isinstance(self.root, str):
            if os.path.isdir(self.root):
                for f in os.listdir(self.root):
                    for line in lines_from_file(os.path.join(self.root, f)):
                        for char in line:
                            yield char
            elif os.path.isfile(self.root):
                for line in lines_from_file(self.root):
                    for char in line:
                        yield char

    def generate(self, indexer, **kwargs):
        """
        Returns:
        --------
        generator (list:int, int) over instances
        """
        if isinstance(self.root, types.GeneratorType):
            for line in self.root:
                for c, l in self._encode_line(line, indexer, **kwargs):
                    yield c, l
        elif isinstance(self.root, str):
            if os.path.isdir(self.root):
                for f in os.listdir(self.root):
                    for l in lines_from_file(os.path.join(self.root, f)):
                        for c, l in self._encode_line(l, indexer, **kwargs):
                            yield c, l
            elif os.path.isfile(self.root):
                for line in lines_from_file(self.root):
                    for c, l in self._encode_line(line, indexer, **kwargs):
                        yield c, l

    def generate_batches(self, indexer, batch_size=128, **kwargs):
        """
        Returns:
        --------
        generator (list:list:int, list:int) over batches of instances
        """
        contexts, labels, n = [], [], 0
        gen = self.generate(indexer, **kwargs)
        while True:
            try:
                context, label = next(gen)
                if n % batch_size == 0 and contexts:
                    yield contexts, labels
                    contexts, labels = [], []
                else:
                    contexts.append(context)
                    labels.append(label)
                    n += 1
            except StopIteration:
                break


if __name__ == '__main__':
    """test"""
    idxr = Indexer(verbose=True)
    from six import moves
    url = 'http://www.gutenberg.org/cache/epub/1342/pg1342.txt'
    text = moves.urllib.request.urlopen(url).read().decode('utf-8')
    train, test = text[:int(len(text) / 8)], text[int(len(text) / 8):]
    idxr.encode_seq(train)
    print("Finished indexing train")
    print("Vocab: %d" % idxr.vocab_len())
    idxr.encode_seq(test)
    print(str(idxr.encoder))
