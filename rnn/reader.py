import collections
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import tensorflow as tf


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
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    oov_count = sum(c for _, c in count_pairs[vocab_size:])
    oov_rate = oov_count / sum(counter.values())
    print('OOV rate: {:.2%}'.format(oov_rate))

    count_pairs = count_pairs[:vocab_size - 1]
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    word_to_id[UNK] = len(word_to_id)
    # TODO - place UNK into correct place by frequncy
    return word_to_id


def _file_to_word_ids(path: Path, word_to_id: Dict[str, int]) -> np.ndarray:
    data = _words_reader(path)
    unk = word_to_id[UNK]
    return np.fromiter((word_to_id.get(word, unk) for word in data),
                       dtype=np.int32, count=-1)


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
        train_data, valid_data, test_data = [
            np.load(str(cached(p))) for p in paths]
    else:
        word_to_id = _build_vocab(train_path, vocab_size)
        train_data, valid_data, test_data = datas = [
            _file_to_word_ids(p, word_to_id) for p in paths]
        for data, path in zip(datas, paths):
            np.save(str(cached(path)), data)
    return train_data, valid_data, test_data


def producer(raw_data, batch_size, num_steps, name=None):
    """ Iterate on the raw data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
        raw_data: one of the raw data outputs from raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.
        name: the name of this operation (optional).

    Returns:
        A pair of Tensors, each shaped [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted
        to the right by one.

    Raises:
        tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, 'Producer', [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(
            raw_data, name='raw_data', dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message='epoch_size == 0, decrease batch_size or num_steps')
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name='epoch_size')

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
        y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
        return x, y
