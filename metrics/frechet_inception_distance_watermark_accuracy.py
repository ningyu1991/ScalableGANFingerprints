# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Frechet Inception Distance (FID)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

class FID_watermark_accuracy(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, E, G, D, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        inception = misc.load_pkl('metrics/inception_v3_features.pkl') # inception_v3_features.pkl
        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)
        accuracy_bit = np.empty([self.num_images], dtype=np.float32)
        accuracy_instance = np.empty([self.num_images], dtype=np.float32)

        # Calculate statistics for reals.
        cache_file = self._get_cache_file_for_reals(num_images=self.num_images)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.isfile(cache_file):
            mu_real, sigma_real = misc.load_pkl(cache_file)
        else:
            for idx, images in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
                begin = idx * minibatch_size
                end = min(begin + minibatch_size, self.num_images)
                activations[begin:end] = inception.run(images[:end-begin], num_gpus=num_gpus, assume_frozen=True)
                if end == self.num_images:
                    break
            mu_real = np.mean(activations, axis=0)
            sigma_real = np.cov(activations, rowvar=False)
            misc.save_pkl((mu_real, sigma_real), cache_file)

        # Construct TensorFlow graph.
        result_expr_fid = []
        result_expr_accuracy_bid = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                E_clone = E.clone()
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shapes[0][1:])
                watermarks = (tf.math.sign(tf.random.uniform([self.minibatch_per_gpu] + Gs_clone.input_shapes[2][1:], minval=-1, maxval=1)) + 1.0) / 2.0
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, labels, watermarks, **Gs_kwargs)
                result_expr_fid.append(inception_clone.get_output_for(tflib.convert_images_to_uint8(images)))
                _, watermark_logits_out = E_clone.get_output_for(images, labels, **Gs_kwargs)
                watermarks_out = (tf.math.sign(watermark_logits_out) + 1.0) / 2.0
                result_expr_accuracy_bid.append(tf.reduce_mean(watermarks*watermarks_out + (1.0-watermarks)*(1.0-watermarks_out), axis=1))

        # Calculate statistics for fakes.
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            end = min(begin + minibatch_size, self.num_images)
            activations[begin:end] = np.concatenate(tflib.run(result_expr_fid), axis=0)[:end-begin]
            accuracy_bit[begin:end] = np.concatenate(tflib.run(result_expr_accuracy_bid), axis=0)[:end-begin]
        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)

        # Calculate FID.
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        self._report_result(np.real(dist), suffix='_fid')
        self._report_result(np.mean(accuracy_bit), suffix='_watermark_bit_accuracy')

#----------------------------------------------------------------------------
