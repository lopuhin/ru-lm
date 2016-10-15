import os.path

import numpy as np
import tensorflow as tf

from . import reader


def test_raw_data():
    string_data = '\n'.join(
        [' hello there i am',
         ' rain as day',
         ' want some cheesy puffs ?'])
    tmpdir = tf.test.get_temp_dir()
    for suffix in 'train', 'valid', 'test':
        filename = os.path.join(tmpdir, '%s.txt' % suffix)
        with tf.gfile.GFile(filename, 'w') as fh:
            fh.write(string_data)
    # Smoke test
    output = reader.raw_data(tmpdir)
    assert len(output) == 4


def test_producer():
    raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
    batch_size = 3
    num_steps = 2
    x, y = reader.producer(raw_data, batch_size, num_steps)
    with tf.Session() as session:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(session, coord=coord)
        try:
            xval, yval = session.run([x, y])
            assert np.allclose(xval, [[4, 3], [5, 6], [1, 0]])
            assert np.allclose(yval, [[3, 2], [6, 1], [0, 3]])
            xval, yval = session.run([x, y])
            assert np.allclose(xval, [[2, 1], [1, 1], [3, 4]])
            assert np.allclose(yval, [[1, 0], [1, 1], [4, 1]])
        finally:
            coord.request_stop()
            coord.join()
