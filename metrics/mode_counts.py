# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Frechet Inception Distance (FID)."""

import os
import glob
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

class mode_counts(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        classifier_pkl = sorted(glob.glob(os.path.join('../classifier/results/stacked_mnist_240k/00000-stylegan2-stacked_mnist_240k-3gpu-config-f_scratch', 'network-snapshot-*.pkl')))[-1]
        classifier = misc.load_pkl(classifier_pkl)

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                classifier_clone = classifier.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, labels, **Gs_kwargs)
                logits = classifier_clone.get_output_for(images, is_training=False)
                result_expr.append(tf.math.argmax(logits, axis=1))

        # Calculate statistics for fakes.
        labels_all = np.empty([self.num_images], dtype=np.float32)
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            end = min(begin + minibatch_size, self.num_images)
            labels_all[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]
        self._report_result(len(np.unique(labels_all)))

#----------------------------------------------------------------------------
