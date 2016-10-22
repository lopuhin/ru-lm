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
    'A type of model. Possible options are: '
    'small, small-sampled, medium, medium-sampled, large.')
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
        self.word_length = config.word_length
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.data = data
        self.name = name


class Model:
    """The model."""

    def __init__(self, is_training: bool, config, input_: Input):
        self.is_training = is_training
        self.input = input_

        num_steps = input_.num_steps
        hidden_size = config.hidden_size
        embedding_size = config.embedding_size or hidden_size
        out_size = config.proj_size or hidden_size
        vocab_size = config.vocab_size
        dtype = data_type()

        self.ys = tf.placeholder(tf.int32, [None, num_steps])
        self.batch_size = tf.placeholder(tf.int32, [])

        self.cnn_inputs = bool(config.cnn_inputs)
        if self.cnn_inputs:
            self.xs = tf.placeholder(
                tf.int32, [None, num_steps, config.word_length])
            inputs = self._get_cnn_inputs(config)
            embedding_out_size = inputs.get_shape()[-1].value
        else:
            self.xs = tf.placeholder(tf.int32, [None, num_steps])
            with tf.device('/cpu:0'):
                embedding = tf.get_variable(
                    'embedding', [vocab_size, embedding_size], dtype=dtype)
                inputs = tf.nn.embedding_lookup(embedding, self.xs)
            embedding_out_size = embedding_size
        tf.add_to_collection('input_xs', self.xs)
        tf.add_to_collection('input_ys', self.ys)
        tf.add_to_collection('batch_size', self.batch_size)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        if hidden_size != embedding_out_size:
            input_proj_w = tf.get_variable(
                'input_proj_w', [embedding_out_size, hidden_size], dtype=dtype)
            input_proj_b = tf.get_variable(
                'input_proj_b', [hidden_size], dtype=dtype)
            inputs = (
                tf.matmul(tf.reshape(inputs, [-1, embedding_out_size]),
                          input_proj_w)
                + input_proj_b)
            inputs = tf.reshape(inputs, [-1, num_steps, hidden_size])

        inputs = [tf.squeeze(input_step, [1])
                  for input_step in tf.split(1, num_steps, inputs)]

        rnn_kwargs = dict(
            num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
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

        self.initial_state = cell.zero_state(self.batch_size, dtype)
        self._save_rnn_state('initial_state', self.initial_state)

        outputs, state = tf.nn.rnn(
            cell, inputs, initial_state=self.initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, out_size])
        tf.add_to_collection('hidden_output', output)
        softmax_w = tf.get_variable(
            'softmax_w', [vocab_size, out_size], dtype=dtype)
        softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype=dtype)
        logits = tf.matmul(output, softmax_w, transpose_b=True) + softmax_b
        softmax = tf.nn.softmax(logits, name='softmax')
        tf.add_to_collection('softmax', softmax)
        labels = tf.reshape(self.ys, [-1])
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits], [labels],
            [tf.ones([self.batch_size * num_steps], dtype=dtype)])
        self.cost = tf.reduce_sum(loss)
        sampled_loss_kwargs = dict(
            weights=softmax_w,
            biases=softmax_b,
            inputs=output,
            labels=tf.expand_dims(labels, 1),
            num_sampled=config.num_sampled,
            num_classes=vocab_size,
            remove_accidental_hits=False,
        )
        if config.sampled_loss == 'nce':
            train_loss = tf.nn.nce_loss(**sampled_loss_kwargs)
            self.train_cost = tf.reduce_mean(train_loss)
        elif config.sampled_loss == 'softmax':
            train_loss = tf.nn.sampled_softmax_loss(**sampled_loss_kwargs)
            self.train_cost = tf.reduce_mean(train_loss)
        else:
            self.train_cost = self.cost
        self.final_state = state
        self._save_rnn_state('final_state', self.final_state)

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

    def _save_rnn_state(self, prefix, state):
        for i, s in enumerate(state):
            tf.add_to_collection('{}_{}_c'.format(prefix, i), s.c)
            tf.add_to_collection('{}_{}_h'.format(prefix, i), s.h)

    def _get_cnn_inputs(self, config):
        dtype = data_type()
        embedding = tf.get_variable(
            'embedding',
            [config.char_vocab_size, config.embedding_size], dtype=dtype)
        inputs = tf.nn.embedding_lookup(embedding, self.xs)
        inputs = tf.expand_dims(
            tf.reshape(inputs, [-1, config.word_length, config.embedding_size]), -1)
        pooled_outputs = []
        for filter_size, num_filters in config.cnn_inputs:
            with tf.variable_scope('Conv-{}'.format(filter_size)):
                filter_shape = [
                    filter_size, config.embedding_size, 1, num_filters]
                conv_w = tf.get_variable('w', filter_shape, dtype=dtype)
                conv_b = tf.get_variable('b', [num_filters], dtype=dtype)
                conv = tf.nn.conv2d(
                    inputs, conv_w, strides=[1, 1, 1, 1], padding='VALID',
                    name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name='relu')
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, config.word_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)
        pooled_output = tf.concat(3, pooled_outputs)
        return tf.reshape(
            pooled_output,
            [-1, config.num_steps, pooled_output.get_shape()[-1].value])


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
    proj_size = None
    embedding_size = None

    sampled_loss = None
    num_sampled = None

    cnn_inputs = None
    char_vocab_size = None
    word_length = None


class SmallConfig(DefaultConfig):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.4
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    embedding_size = 128
    hidden_size = 256
    proj_size = 128
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 32
    vocab_size = 10000


class SmallSampledConfig(SmallConfig):
    """Small sampled loss config."""
    sampled_loss = 'softmax'
    num_sampled = 1024


class SmallCNNConfig(SmallConfig):
    embedding_size = 32
    cnn_inputs = [(2, 100), (3, 100), (4, 100), (5, 100), (6, 100)]
    char_vocab_size = 128
    word_length = 64


class MediumConfig(DefaultConfig):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 0.2
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    embedding_size = 512
    hidden_size = 2048
    proj_size = 512
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.8
    batch_size = 128
    vocab_size = 150000


class MediumSampledConfig(MediumConfig):
    """Medium sampled config."""
    sampled_loss = 'softmax'
    num_sampled = 4092
    vocab_size = 300000


class LargeConfig(DefaultConfig):
    """Large config."""
    init_scale = 0.05
    learning_rate = 0.2
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    embedding_size = 768
    hidden_size = 3076
    proj_size = 768
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 1.0
    lr_decay = 1 / 1.15
    batch_size = 128
    vocab_size = 150000


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
    batch_size = model.input.batch_size
    step_size = batch_size * model.input.num_steps

    fetches = {
        'cost': model.train_cost if model.is_training else model.cost,
        'final_state': model.final_state,
    }
    if model.is_training:
        fetches['eval_op'] = model.train_op
    state = session.run(model.initial_state, {model.batch_size: batch_size})

    for batch_start in range(0, len(data) - 1, step_size):
        xs = data[batch_start: batch_start + step_size]
        ys = data[batch_start + 1: batch_start + step_size + 1]
        if len(ys) < step_size:
            break  # skip last incomplete batch
        if model.cnn_inputs:
            ys = np.reshape(ys[:, -1], [batch_size, model.input.num_steps])
            xs = np.reshape(
                xs[:, :-1],
                [batch_size, model.input.num_steps, model.input.word_length])
        else:
            xs, ys = [
                np.reshape(it, [batch_size, model.input.num_steps])
                for it in [xs, ys]]
        feed_dict = {model.xs: xs, model.ys: ys, model.batch_size: batch_size}
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
        'small-sampled': SmallSampledConfig(),
        'small-cnn': SmallCNNConfig(),
        'medium': MediumConfig(),
        'medium-sampled': MediumSampledConfig(),
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

    train_data, valid_data, test_data = reader.load_raw_data(
        data_path=Path(FLAGS.data_path),
        vocab_size=config.vocab_size,
        char_vocab_size=config.char_vocab_size,
        word_length=config.word_length)

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
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.allow_growth = True
        with sv.managed_session(config=tf_config) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print('Epoch: {} Learning rate: {:.3f}'
                      .format(i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, verbose=True)
                print('Epoch: {} Train Perplexity: {:.3f}'
                      .format(i + 1, train_perplexity))

                if FLAGS.save_path:
                    save_path = str(Path(FLAGS.save_path) / 'model')
                    print('Saving model to {}.'.format(save_path))
                    sv.saver.save(session, save_path, global_step=sv.global_step)

                valid_perplexity = run_epoch(session, mvalid)
                print('Epoch: {} Valid Perplexity: {:.3f}'
                      .format(i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print('Test Perplexity: {:.3f}'.format(test_perplexity))


def run():
    tf.app.run(main=main)
