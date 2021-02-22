# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Pairwise LPIPS (PairLPIPS)."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

class PairLPIPS(metric_base.MetricBase):
    def __init__(self, num_samples, crop, minibatch_per_gpu, Gs_overrides, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.crop = crop
        self.minibatch_per_gpu = minibatch_per_gpu
        self.Gs_overrides = Gs_overrides

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        Gs_kwargs = dict(Gs_kwargs)
        Gs_kwargs.update(self.Gs_overrides)
        minibatch_size = num_gpus * self.minibatch_per_gpu

        # Construct TensorFlow graph.
        distance_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                noise_vars = [var for name, var in Gs_clone.components.synthesis.vars.items() if name.startswith('noise')]

                # Generate random latents and interpolation t-values.
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)

                # Synthesize images.
                with tf.control_dependencies([var.initializer for var in noise_vars]): # use same noise inputs for the entire minibatch
                    images = Gs_clone.get_output_for(latents, labels, randomize_noise=False, **Gs_kwargs)
                    images = tf.cast(images, tf.float32)

                # Crop only the face region.
                if self.crop:
                    c = int(images.shape[2] // 8)
                    images = images[:, :, c*3 : c*7, c*2 : c*6]

                # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
                factor = images.shape[2] // 256
                if factor > 1:
                    images = tf.reshape(images, [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
                    images = tf.reduce_mean(images, axis=[3,5])

                # Scale dynamic range from [-1,1] to [0,255] for VGG.
                images = (images + 1) * (255 / 2)

                # Evaluate perceptual distance.
                distance_measure = misc.load_pkl('metrics/vgg16_zhang_perceptual.pkl') # vgg16_zhang_perceptual.pkl
                distance_expr.append(distance_measure.get_output_for(images[0::2], images[1::2]))

        # Sampling loop.
        all_distances = []
        for begin in range(0, self.num_samples, minibatch_size):
            self._report_progress(begin, self.num_samples)
            all_distances += tflib.run(distance_expr)
        all_distances = np.concatenate(all_distances, axis=0)
        self._report_result(np.mean(all_distances))

#----------------------------------------------------------------------------
