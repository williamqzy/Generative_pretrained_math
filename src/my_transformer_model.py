import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

class MyTransformerModel:
    def __init__(self):
        pass

    @staticmethod
    def default_hyperparameters():
        return HParams(
            vocabulary_size=0,
            context_size=1024,
            embedding_size=768,
            num_heads=12,
            num_layers=12,
        )

    @staticmethod
    def shape_list(x):
        """Deal with dynamic shape in TensorFlow cleanly."""
        static = x.shape.as_list()
        dynamic = tf.shape(x)
        return [dynamic[i] if s is None else s for i, s in enumerate(static)]

    @staticmethod
    def softmax(x, axis=-1):
        x = x - tf.reduce_max(x, axis=axis, keepdims=True)
        ex = tf.exp(x)
        return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

    @staticmethod
    def layer_norm(x, scope, *, axis=-1, epsilon=1e-5):
        """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
        with tf.variable_scope(scope):
            n_state = x.shape[-1].value
            g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
            b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
            u = tf.reduce_mean(x, axis=axis, keepdims=True)
            s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
            x = (x - u) * tf.rsqrt(s + epsilon)
            x = x * g + b
            return x

    @staticmethod
    def split_states(x, n):
        """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
        *start, m = MyTransformerModel.shape_list(x)
        return tf.reshape(x, start + [n, m // n])

    @staticmethod
    def merge_states(x):
        """Smash the last two dimensions of x into a single dimension."""
        *start, a, b = MyTransformerModel.shape_list(x)
        return tf.reshape(x, start + [a * b])

    @staticmethod
    def conv1d(x, scope, nf, *, w_init_stdev=0.02):
        with tf.variable_scope(scope):
            *start, nx = MyTransformerModel.shape_list(x)
            w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b, start + [nf])
            return c

    @staticmethod
    def attention_mask(nd, ns, *, dtype):
        """1's in the lower triangle, counting from the lower right corner.

        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    @staticmethod
    def masked_attention_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = MyTransformerModel.shape_list(w)
        b = MyTransformerModel.attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
        return w

    @staticmethod
    def multihead_attention(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))
        w = MyTransformerModel.masked_attention_weights(w)
        w = MyTransformerModel.softmax(w)
        a = tf.matmul(w, v)
        return a

    @staticmethod
    def mlp(x, scope, n_state, *, hyperparameters):
        with tf.variable_scope(scope):
            nx = x.shape[-1].value
            h = MyTransformerModel.gelu(MyTransformerModel.conv1d(x, 'c_fc', n_state))
            h2 = MyTransformerModel.conv1d(h, 'c_proj', nx)
            return h2

    @staticmethod
    def block(x, scope, *, past, hyperparameters):
        with tf.variable_scope(scope):
            nx = x.shape[-1].value
            a, present = MyTransformerModel.attn(MyTransformerModel.layer_norm(x, 'ln_1'), 'attn', nx, past=past,
                                                hyperparameters=hyperparameters)
            x = x + a
            m = MyTransformerModel.mlp(MyTransformerModel.layer_norm(x, 'ln_2'), 'mlp', nx * 4,
                                       hyperparameters=hyperparameters)
            x = x + m
            return x, present

    @staticmethod
    def past_shape(*, hyperparameters, batch_size=None, sequence=None):
        return [batch_size, hyperparameters.num_layers, 2, hyperparameters.num_heads, sequence,
                hyperparameters.embedding_size // hyperparameters.num_heads]

    @staticmethod
    def expand_tile(value, size):
        """Add a new axis of given size."""
        value = tf.convert_to_tensor(value, name='value')
        ndims = value.shape.ndims
        return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)

    @staticmethod
    def positions_for(tokens, past_length):
        batch_size = tf.shape(tokens)[0]
        nsteps = tf.shape(tokens)[1]
        return MyTransformerModel.expand_tile(past_length + tf.range(nsteps), batch_size)

    def model(self, hyperparameters, X, past=None, scope='model', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            results = {}
            batch, sequence = MyTransformerModel.shape_list(X)

            wpe = tf.get_variable('wpe', [hyperparameters.context_size, hyperparameters.embedding_size],
                                 initializer=tf.random_normal_initializer(stddev=0.01))
            wte = tf.get_variable('wte', [hyperparameters.vocabulary_size, hyperparameters.embedding_size],
                                 initializer=tf.random_normal_initializer(stddev=0.02))
            past_length = 0 if past is None else tf.shape(past)[-2]
            h = tf.gather(wte, X) + tf.gather(wpe, MyTransformerModel.positions_for(X, past_length))

            # Transformer
            presents = []
            pasts = tf.unstack(past, axis=1) if past is not None else [None] * hyperparameters.num_layers
            assert len(pasts) == hyperparameters.num_layers
            for layer, past in enumerate(pasts):
                h, present = MyTransformerModel.block(h, 'h%d' % layer, past=past, hyperparameters=hyperparameters)
                presents.append(present)
            results['present'] = tf.stack(presents, axis=1)
            h = MyTransformerModel.layer_norm(h, 'ln_f')

            # Language model loss.  Do tokens <n predict token n?
            h_flat = tf.reshape(h, [batch * sequence, hyperparameters.embedding_size])
            logits = tf.matmul(h_flat, wte, transpose_b=True)
            logits = tf.reshape(logits, [batch, sequence, hyperparameters.vocabulary_size])
            results['logits'] = logits
            return results
