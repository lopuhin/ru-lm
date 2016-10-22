from pathlib import Path

import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, path: Path):
        self.session = tf.Session()
        saver = tf.train.import_meta_graph('{}.meta'.format(path))
        saver.restore(self.session, str(path))
        self.input_xs = tf.get_collection('input_xs')[0]
        self.batch_size = tf.get_collection('batch_size')[0]
        self.softmax = tf.get_collection('softmax')[0]

    def predict(self):
        xs = np.zeros([1, 20])
        xs[0, :3] = [1, 2, 3]
        softmax = self.session.run(
            self.softmax, feed_dict={self.input_xs: xs, self.batch_size: 1})
        return softmax
