from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from .reader import load_vocab, tokens_to_ids


class Model:
    def __init__(self, model_path: Path, vocab_path: Path):
        self.word_to_id = load_vocab(vocab_path)
        self.id_to_word = {id_: word for word, id_ in self.word_to_id.items()}
        self.session = tf.Session()
        saver = tf.train.import_meta_graph('{}.meta'.format(model_path))
        saver.restore(self.session, str(model_path))
        self.input_xs = tf.get_collection('input_xs')[0]
        self.batch_size = tf.get_collection('batch_size')[0]
        self.softmax = tf.get_collection('softmax')[0]
        self.num_steps = 20

    def predict(self, tokens: List[str]) -> np.ndarray:
        tokens = tokens[-self.num_steps:]
        xs = np.zeros([1, self.num_steps])
        n = len(tokens)
        xs[0, :n] = list(tokens_to_ids(tokens, self.word_to_id))
        return self.session.run(
            self.softmax[n - 1],
            feed_dict={self.input_xs: xs, self.batch_size: 1})

    def predict_top(self, tokens: List[str], top=10) -> List[Tuple[str, float]]:
        probs = self.predict(tokens)
        top_indices = argsort_k_largest(probs, top)
        return [(self.id_to_word[id_], probs[id_]) for id_ in top_indices]

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.__exit__(None, None, None)


def argsort_k_largest(x, k):
    if k >= len(x):
        return np.argsort(x)[::-1]
    indices = np.argpartition(x, -k)[-k:]
    values = x[indices]
    return indices[np.argsort(-values)]
