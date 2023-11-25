import fire
import json
import os
import numpy as np
import tensorflow as tf

import my_model as my_model_module  # Renamed the model module to 'my_model_module'
import my_sample as my_sample_module  # Renamed the sample module to 'my_sample_module'
import my_encoder as my_encoder_module  # Renamed the encoder module to 'my_encoder_module'

class TextSampler:
    def __init__(self, model_name='124M', seed=None, nsamples=0, batch_size=1, length=None,
                 temperature=1, top_k=0, top_p=1, models_dir='models'):
        """
        Initialize the TextSampler.
        :model_name: String, which model to use
        :seed: Integer seed for random number generators, fix seed to reproduce results
        :nsamples: Number of samples to return, if 0, continues to generate samples indefinitely.
        :batch_size: Number of batches (only affects speed/memory).
        :length: Number of tokens in generated text, if None (default), is determined by model hyperparameters
        :temperature: Float value controlling randomness in Boltzmann distribution.
        :top_k: Integer value controlling diversity.
        :top_p: Float value controlling diversity.
        :models_dir: Path to the parent folder containing model subfolders.
        """
        self.model_name = model_name
        self.seed = seed
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.length = length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.models_dir = models_dir

    def run_sampler(self):
        """
        Run the TextSampler.
        """
        models_dir = os.path.expanduser(os.path.expandvars(self.models_dir))
        enc = my_encoder_module.get_encoder(self.model_name, models_dir)
        hparams = my_model_module.default_hparams()

        with open(os.path.join(models_dir, self.model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if self.length is None:
            self.length = hparams.n_ctx
        elif self.length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        with tf.Session(graph=tf.Graph()) as sess:
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)

            output = my_sample_module.sample_sequence(
                hparams=hparams, length=self.length,
                start_token=enc.encoder[''],
                batch_size=self.batch_size,
                temperature=self.temperature, top_k=self.top_k, top_p=self.top_p
            )[:, 1:]

            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, self.model_name))
            saver.restore(sess, ckpt)

            generated = 0
            while self.nsamples == 0 or generated < self.nsamples:
                out = sess.run(output)
                for i in range(self.batch_size):
                    generated += self.batch_size
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)

if __name__ == '__main__':
    fire.Fire(TextSampler)
