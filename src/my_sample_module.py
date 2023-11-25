import tensorflow as tf

import my_model_module as my_model  # Renamed the model module to 'my_model'
from my_model_module import HParams  # Import HParams from the model module

class MySampler:
    def __init__(self, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
        self.hparams = hparams
        self.length = length
        self.start_token = start_token
        self.batch_size = batch_size
        self.context = context
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    @staticmethod
    def top_k_logits(logits, k):
        if k == 0:
            # no truncation
            return logits

        def _top_k():
            values, _ = tf.nn.top_k(logits, k=k)
            min_values = values[:, -1, tf.newaxis]
            return tf.where(
                logits < min_values,
                tf.ones_like(logits, dtype=logits.dtype) * -1e10,
                logits,
            )

        return tf.cond(
            tf.equal(k, 0),
            lambda: logits,
            lambda: _top_k(),
        )

    @staticmethod
    def top_p_logits(logits, p):
        """Nucleus sampling"""
        batch, _ = logits.shape.as_list()
        sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
        indices = tf.stack([
            tf.range(0, batch),
            # number of indices to include
            tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
        ], axis=-1)
        min_values = tf.gather_nd(sorted_logits, indices)
        return tf.where(
            logits < min_values,
            tf.ones_like(logits) * -1e10,
            logits,
        )

    def sample_sequence(self):
        if self.start_token is None:
            assert self.context is not None, 'Specify exactly one of start_token and context!'
        else:
            assert self.context is None, 'Specify exactly one of start_token and context!'
            self.context = tf.fill([self.batch_size, 1], self.start_token)

        def step(tokens, past=None):
            lm_output = my_model.model(hparams=self.hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

            logits = lm_output['logits'][:, :, :self.hparams.n_vocab]
            presents = lm_output['present']
            presents.set_shape(my_model.past_shape(hparams=self.hparams, batch_size=self.batch_size))
            return {
                'logits': logits,
                'presents': presents,
            }

        with tf.name_scope('sample_sequence'):
            def body(past, prev, output):
                next_outputs = step(prev, past=past)
                logits = next_outputs['logits'][:, -1, :] / tf.to_float(self.temperature)
                logits = self.top_k_logits(logits, k=self.top_k)
                logits = self.top_p_logits(logits, p=self.top_p)
                samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
                return [
                    next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                    samples,
                    tf.concat([output, samples], axis=1)
                ]

            past, prev, output = body(None, self.context, self.context)

            def cond(*args):
                return True

            _, _, tokens = tf.while_loop(
                cond=cond, body=body,
                maximum_iterations=self.length - 1,
                loop_vars=[
                    past,
                    prev,
                    output
                ],
                shape_invariants=[
                    tf.TensorShape(my_model.past_shape(hparams=self.hparams, batch_size=self.batch_size)),
                    tf.TensorShape([self.batch_size, None]),
                    tf.TensorShape([self.batch_size, None]),
                ],
                back_prop=False,
            )

            return tokens
