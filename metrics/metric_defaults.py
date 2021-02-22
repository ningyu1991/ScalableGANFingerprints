# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Default metric definitions."""

from dnnlib import EasyDict

#----------------------------------------------------------------------------

metric_defaults = EasyDict([(args.name, args) for args in [
    #------------------------ watermark ------------------------
    EasyDict(name='fid_watermark_accuracy_50k',    func_name='metrics.frechet_inception_distance_watermark_accuracy.FID_watermark_accuracy', num_images=50000, minibatch_per_gpu=32),
    #------------------------ Stacked MNIST ------------------------
    EasyDict(name='mode_counts_24k',    func_name='metrics.mode_counts.mode_counts', num_images=24000, minibatch_per_gpu=32),
    EasyDict(name='KL24k',    func_name='metrics.KL.KL', num_images=24000, minibatch_per_gpu=32),
	#------------------------ IMLE+GAN ------------------------
	#EasyDict(name='fid3k',    func_name='metrics.frechet_inception_distance.FID', num_images=3000, minibatch_per_gpu=8),
    EasyDict(name='pr3k3',    func_name='metrics.precision_recall.PR',            num_images=3000, nhood_size=3, minibatch_per_gpu=8, row_batch_size=10000, col_batch_size=10000),
    EasyDict(name='pr10k3',    func_name='metrics.precision_recall.PR',            num_images=10000, nhood_size=3, minibatch_per_gpu=8, row_batch_size=10000, col_batch_size=10000),
    EasyDict(name='pr30k3',    func_name='metrics.precision_recall.PR',            num_images=30000, nhood_size=3, minibatch_per_gpu=8, row_batch_size=10000, col_batch_size=10000),
    EasyDict(name='pair_lpips_50k', func_name='metrics.pairwise_lpips.PairLPIPS',     num_samples=50000, crop=True, minibatch_per_gpu=8, Gs_overrides=dict(dtype='float32', mapping_dtype='float32')),
    #------------------------ official original ------------------------
    EasyDict(name='fid50k',    func_name='metrics.frechet_inception_distance.FID', num_images=50000, minibatch_per_gpu=32),
    EasyDict(name='is50k',     func_name='metrics.inception_score.IS',             num_images=50000, num_splits=10, minibatch_per_gpu=8),
    EasyDict(name='ppl_zfull', func_name='metrics.perceptual_path_length.PPL',     num_samples=50000, epsilon=1e-4, space='z', sampling='full', crop=True, minibatch_per_gpu=4, Gs_overrides=dict(dtype='float32', mapping_dtype='float32')),
    EasyDict(name='ppl_wfull', func_name='metrics.perceptual_path_length.PPL',     num_samples=50000, epsilon=1e-4, space='w', sampling='full', crop=True, minibatch_per_gpu=4, Gs_overrides=dict(dtype='float32', mapping_dtype='float32')),
    EasyDict(name='ppl_zend',  func_name='metrics.perceptual_path_length.PPL',     num_samples=50000, epsilon=1e-4, space='z', sampling='end', crop=True, minibatch_per_gpu=4, Gs_overrides=dict(dtype='float32', mapping_dtype='float32')),
    EasyDict(name='ppl_wend',  func_name='metrics.perceptual_path_length.PPL',     num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=True, minibatch_per_gpu=4, Gs_overrides=dict(dtype='float32', mapping_dtype='float32')),
    EasyDict(name='ppl2_wend', func_name='metrics.perceptual_path_length.PPL',     num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, minibatch_per_gpu=4, Gs_overrides=dict(dtype='float32', mapping_dtype='float32')),
    EasyDict(name='ls',        func_name='metrics.linear_separability.LS',         num_samples=200000, num_keep=100000, attrib_indices=range(40), minibatch_per_gpu=4),
    EasyDict(name='pr50k3',    func_name='metrics.precision_recall.PR',            num_images=50000, nhood_size=3, minibatch_per_gpu=8, row_batch_size=10000, col_batch_size=10000),
]])

#----------------------------------------------------------------------------