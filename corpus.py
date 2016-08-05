# coding: utf-8

import os
import types
import pickle as p  # python3
import json


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


def indexer_from_dict(d):
    indexer = Indexer()
    indexer.encoder = d['encoder']
    indexer.decoder = {int(k): v for (k, v) in d['decoder']}
    return indexer


class Indexer(object):
    def __init__(self, reserved=None):
        """
        Parameters:
        -----------
        reserved: dict {int: any}, preindexed reserved items.
        This is useful in case of reserved values (e.g. padding).
        Indices must start at 0.

        Example:
        --------
        indexer = Indexer(reserved={0: 'padding', 1: 'OOV'})
        """
        self.decoder = {}
        self.encoder = {}
        if reserved:
            if sorted(reserved) != list(range(len(reserved))):
                raise ValueError("reserved must start at 0")
            self._current = len(reserved)
            self._extra = reserved
        else:
            self._current = 0
            self._extra = {}

    def vocab(self):
        return self.encoder.keys()

    def vocab_len(self):
        return len(self.encoder)

    def encode(self, s, oov_idx=None):
        """
        Parameters:
        -----------
        s: object, object to index
        oov_idx: int or None, if None new indices are assigned to previously
        unseen items (OOVs), if int oov_idx is returned for OOVs

        Returns:
        --------
        idx (int)
        """
        if s in self.encoder:
            return self.encoder[s]
        elif oov_idx:
            return oov_idx
        else:
            idx = self._current
            self.encoder[s] = idx
            self.decoder[idx] = s
            self._current += 1
            return idx

    def decode(self, idx):
        if idx not in self.decoder and idx not in self._extra:
            raise ValueError("Cannot found index [%d]" % idx)
        try:
            return self.decoder[idx]
        except KeyError:
            return self._extra[idx]

    def encode_seq(self, seq):
        return [self.encode(x) for x in seq]

    def _to_json(self):
        return {'encoder': self.encoder, 'decoder': self.decoder}

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
                return indexer_from_dict(json.load(f))
        else:
            raise ValueError('Unrecognized mode %s' % mode)


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
                left = pad(encoded_line[minidx: idx],
                           self.context, paddir='left')
            if self.side in {'right', 'both'}:
                right = pad(encoded_line[idx + 1: maxidx],
                            self.context, paddir='right')
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
