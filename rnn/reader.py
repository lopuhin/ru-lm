import collections
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


EOS = '<eos>'
UNK = '<unk>'


def _words_reader(path: Path) -> Iterable[str]:
    # TODO - progress indicator
    print('Reading {}'.format(path))
    with path.open('rt', encoding='utf8') as f:
        for line in f:
            yield from line.replace('\n', '<eos>').split()
    print('done.')


def _build_vocab(path: Path, vocab_size: int) -> Dict[str, int]:
    counter = collections.Counter(_words_reader(path))
    sort_key = lambda x: (-x[1], x[0])
    count_pairs = sorted(counter.items(), key=sort_key)

    oov_count = sum(c for _, c in count_pairs[vocab_size:])
    oov_rate = oov_count / sum(counter.values())
    print('OOV rate: {:.2%}'.format(oov_rate))

    if oov_count:
        count_pairs = count_pairs[:vocab_size - 1]
        count_pairs.append((UNK, oov_count))
        count_pairs.sort(key=sort_key)
    else:
        count_pairs = count_pairs[:vocab_size]
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _save_vocab(path: Path, word_to_id: Dict[str, int]):
    with path.open('wt') as f:
        json.dump(word_to_id, f,
                  ensure_ascii=False, sort_keys=True, indent=True)


def load_vocab(path) -> Dict[str, int]:
    with path.open('rt') as f:
        return json.load(f)


def _file_to_word_ids(path: Path, word_to_id: Dict[str, int]) -> np.ndarray:
    data = _words_reader(path)
    return np.fromiter(tokens_to_ids(data, word_to_id), dtype=np.int32, count=-1)


def tokens_to_ids(tokens: Iterable[str], word_to_id: Dict[str, int])\
        -> Iterable[int]:
    unk = word_to_id.get(UNK)
    return (word_to_id.get(tokens, unk) for tokens in tokens)


def load_raw_data(data_path: Path, vocab_size: int):
    """ Load raw data from data directory 'data_path'.

    Reads text files, converts strings to integer ids.

    Args:
        data_path: string path to the directory where
            train.txt, valid.txt and test.txt are located.
        vocab_size: size of vocabulary (includes EOS and UNK)

    Returns:
        tuple (train_data, valid_data, test_data)
        where each of the data objects can be passed to Iterator.
    """
    cached = lambda path: Path('{}.{}.npy'.format(path, vocab_size))
    train_path = data_path / 'train.txt'
    valid_path = data_path / 'valid.txt'
    test_path = data_path / 'test.txt'
    paths = [train_path, valid_path, test_path]
    if all(cached(p).exists() for p in paths):
        print('Loading cached corpus')
        train_data, valid_data, test_data = [
            np.load(str(cached(p)), mmap_mode='r') for p in paths]
        print('done.')
    else:
        word_to_id = _build_vocab(train_path, vocab_size)
        train_data, valid_data, test_data = datas = [
            _file_to_word_ids(p, word_to_id) for p in paths]
        for data, path in zip(datas, paths):
            np.save(str(cached(path)), data)
        _save_vocab(data_path / ('vocab-{}.json'.format(vocab_size)), word_to_id)
    return train_data, valid_data, test_data
