from collections import deque
from pathlib import Path

import numpy as np
import progressbar
import tensorflow as tf

from rnn import reader


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    'model', 'small',
    'A type of model. Possible options are: small, medium, medium-nce, large.')
flags.DEFINE_string(
    'data_path', None,
    'Where the training/test data is stored.')
flags.DEFINE_string(
    'save_path', None,
    'Model output directory.')
flags.DEFINE_bool(
    'use_fp16', False,
    'Train using 16-bit floats instead of 32bit floats')

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class Input:
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.data = data
        self.name = name


class Model:
    """The model."""

    def __init__(self, is_training: bool, config, input_: Input):
        self.is_training = is_training
        self.input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        out_size = config.proj_size or size
        vocab_size = config.vocab_size

        rnn_kwargs = dict(
            num_units=size, forget_bias=1.0, state_is_tuple=True)
        if config.proj_size:
            lstm_cell = tf.nn.rnn_cell.LSTMCell(
                num_proj=config.proj_size, **rnn_kwargs)
        else:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(**rnn_kwargs)

        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell] * config.num_layers, state_is_tuple=True)

        self.initial_state = cell.zero_state(batch_size, data_type())

        self.xs = tf.placeholder(tf.int32, [None, num_steps])
        self.ys = tf.placeholder(tf.int32, [None, num_steps])

        with tf.device('/cpu:0'):
            embedding = tf.get_variable(
                'embedding', [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, self.xs)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(input_step, [1])
                  for input_step in tf.split(1, num_steps, inputs)]
        outputs, state = tf.nn.rnn(
            cell, inputs, initial_state=self.initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, out_size])
        softmax_w = tf.get_variable(
            'softmax_w', [vocab_size, out_size], dtype=data_type())
        softmax_b = tf.get_variable(
            'softmax_b', [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w, transpose_b=True) + softmax_b
        labels = tf.reshape(self.ys, [-1])
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits], [labels],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self.cost = tf.reduce_sum(loss)
        if config.use_nce:
            train_loss = tf.add_n(
                [tf.nn.nce_loss(
                    softmax_w, softmax_b,
                    inputs=out,
                    labels=tf.expand_dims(self.ys[:, idx], 1),
                    num_sampled=batch_size * 32,
                    num_classes=vocab_size,
                ) for idx, out in enumerate(outputs)])
            # FIXME - don't we need to divide train_loss by len(outputs) ?
            self.train_cost = tf.reduce_sum(train_loss) / batch_size
        else:
            self.train_cost = self.cost
        self.final_state = state

        if not self.is_training:
            return

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.train_cost, tvars)
        if config.max_grad_norm:
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.train_cost, tvars), config.max_grad_norm)
        optimizer = tf.train.AdagradOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(self.lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


"""
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
"""


class DefaultConfig:
    use_nce = False
    proj_size = None


class SmallConfig(DefaultConfig):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.4
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    hidden_size = 256
    proj_size = 128
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 32
    vocab_size = 10000


class SmallNCEConfig(SmallConfig):
    """Small NCE config."""
    use_nce = True


class MediumConfig(DefaultConfig):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 0.2
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    hidden_size = 2024
    proj_size = 512
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.8
    batch_size = 128
    vocab_size = 150000


class MediumNCEConfig(MediumConfig):
    """Medium NCE config."""
    use_nce = True


class LargeConfig(DefaultConfig):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 2048
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.8
    lr_decay = 1 / 1.15
    batch_size = 128
    vocab_size = 200000


class TestConfig:
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session: tf.Session, model: Model, verbose: bool=False):
    """Runs the model on the given data."""
    data = model.input.data
    bar = make_progressbar(len(data)) if verbose else None
    total_costs = 0.
    total_iters = 0
    costs = deque(maxlen=100)
    step_size = model.input.batch_size * model.input.num_steps

    fetches = {
        'cost': model.train_cost if model.is_training else model.cost,
        'final_state': model.final_state,
    }
    if model.is_training:
        fetches['eval_op'] = model.train_op
    state = session.run(model.initial_state)

    for batch_start in range(0, len(data) - 1, step_size):
        xs = data[batch_start: batch_start + step_size]
        ys = data[batch_start + 1: batch_start + step_size + 1]
        if len(ys) < step_size:
            break  # skip last incomplete batch
        xs, ys = [
            np.reshape(it, [model.input.batch_size, model.input.num_steps])
            for it in [xs, ys]]
        feed_dict = {model.xs: xs, model.ys: ys}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches, feed_dict)
        cost = vals['cost']
        state = vals['final_state']

        costs.append(cost)
        total_costs += cost
        total_iters += step_size
        if bar is not None:
            bar.update(total_iters, pplx=np.exp(np.mean(costs) / step_size))

    if verbose:
        bar.finish()
    return np.exp(total_costs / total_iters)


def make_progressbar(max_value: int):
    return progressbar.ProgressBar(
        max_value=max_value,
        widgets=[
            progressbar.DynamicMessage('pplx'), ', ',
            progressbar.FileTransferSpeed(unit='it', prefixes=['']), ', ',
            progressbar.SimpleProgress(), ',',
            progressbar.Percentage(), ' ',
            progressbar.Bar(), ' ',
            progressbar.AdaptiveETA(),
        ]).start()


def get_config():
    return {
        'small': SmallConfig(),
        'small-nce': SmallNCEConfig(),
        'medium': MediumConfig(),
        'medium-nce': MediumNCEConfig(),
        'large': LargeConfig(),
        'test': TestConfig(),
    }[FLAGS.model]


def main(_):
    if not FLAGS.data_path:
        raise ValueError('Must set --data_path to data directory')

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    train_data, valid_data, test_data = \
        reader.load_raw_data(Path(FLAGS.data_path), config.vocab_size)

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        with tf.name_scope('Train'):
            train_input = Input(
                config=config, data=train_data, name='TrainInput')
            with tf.variable_scope(
                    'Model', reuse=None, initializer=initializer):
                m = Model(is_training=True, config=config, input_=train_input)
            tf.scalar_summary('Training Loss', m.cost)
            tf.scalar_summary('Learning Rate', m.lr)

        with tf.name_scope('Valid'):
            valid_input = Input(
                config=config, data=valid_data, name='ValidInput')
            with tf.variable_scope(
                    'Model', reuse=True, initializer=initializer):
                mvalid = Model(
                    is_training=False, config=config, input_=valid_input)
            tf.scalar_summary('Validation Loss', mvalid.cost)

        with tf.name_scope('Test'):
            test_input = Input(config=config, data=test_data, name='TestInput')
            with tf.variable_scope(
                    'Model', reuse=True, initializer=initializer):
                mtest = Model(
                    is_training=False, config=eval_config, input_=test_input)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print('Epoch: {} Learning rate: {:.3f}'
                      .format(i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, verbose=True)
                print('Epoch: {} Train Perplexity: {:.3f}'
                      .format(i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print('Epoch: {} Valid Perplexity: {:.3f}'
                      .format(i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print('Test Perplexity: {:.3f}'.format(test_perplexity))

            if FLAGS.save_path:
                print('Saving model to {}.'.format(FLAGS.save_path))
                sv.saver.save(
                    session, FLAGS.save_path, global_step=sv.global_step)


def run():
    tf.app.run(main=main)
