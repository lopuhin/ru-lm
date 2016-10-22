import collections
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


EOS = '<eos>'
UNK = '<unk>'


def _words_reader(path: Path) -> Iterable[str]:
    print('Reading words from {}'.format(path))
    with path.open('rt', encoding='utf8') as f:
        for line in f:
            yield from line.replace('\n', '<eos>').split()
    print('done.')


def _chars_reader(path: Path) -> Iterable[str]:
    print('Reading chars from {}'.format(path))
    with path.open('rt', encoding='utf8') as f:
        for line in f:
            # Space is included, it is used for padding
            yield from line.strip()
    print('done.')


def _build_vocab(path: Path, vocab_size: int, reader=_words_reader)\
        -> Dict[str, int]:
    counter = collections.Counter(reader(path))
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


def _file_to_char_word_ids(
        path: Path, word_to_id: Dict[str, int],
        char_to_id: Dict[str, int], word_length: int) -> np.ndarray:
    word_unk = word_to_id.get(UNK)
    char_unk = char_to_id.get(UNK)

    def reader():
        end_padding = ' ' * word_length
        for word in _words_reader(path):
            word_id = word_to_id.get(word, word_unk)
            for c in (' ' + word + end_padding)[:word_length]:
                yield char_to_id.get(c, char_unk)
            yield word_id

    return (np.fromiter(reader(), dtype=np.int32, count=-1)
            .reshape([-1, word_length + 1]))


def tokens_to_ids(tokens: Iterable[str], word_to_id: Dict[str, int])\
        -> Iterable[int]:
    unk = word_to_id.get(UNK)
    return (word_to_id.get(tokens, unk) for tokens in tokens)


def load_raw_data(data_path: Path, vocab_size: int,
                  char_vocab_size: int=None, word_length: int=None):
    """ Load raw data from data directory 'data_path'.

    Reads text files, converts strings to integer ids.

    Args:
        data_path: string path to the directory where
            train.txt, valid.txt and test.txt are located.
        vocab_size: size of vocabulary (includes EOS and UNK)
        char_vocab_size: size of character vocabulary, if using CNN inputs
        word_length: max word length, must be specified for CNN inputs

    Returns:
        tuple (train_data, valid_data, test_data)
        where each of the data objects can be passed to Input.
    """
    cnn_inputs = char_vocab_size is not None
    if cnn_inputs:
        cached = lambda p: Path(
            '{}.{}.{}-{}.npy'.format(p, vocab_size, char_vocab_size, word_length))
    else:
        cached = lambda p: Path('{}.{}.npy'.format(p, vocab_size))
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
        char_to_id = None
        if cnn_inputs:
            char_to_id = _build_vocab(
                train_path, char_vocab_size, reader=_chars_reader)
            data_arrays = [
                _file_to_char_word_ids(p, word_to_id, char_to_id, word_length)
                for p in paths]
        else:
            data_arrays = [_file_to_word_ids(p, word_to_id) for p in paths]
        train_data, valid_data, test_data = data_arrays
        for data, path in zip(data_arrays, paths):
            np.save(str(cached(path)), data)
        _save_vocab(
            data_path / ('vocab-{}.json'.format(vocab_size)), word_to_id)
        if cnn_inputs:
            _save_vocab(
                data_path / ('vocab-char-{}.json'.format(char_vocab_size)),
                char_to_id)
    return train_data, valid_data, test_data
