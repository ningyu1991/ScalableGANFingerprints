# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import copy
import os
import sys
import numpy as np

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults

#----------------------------------------------------------------------------

_valid_configs = [
    # Table 1
    'config-a', # Baseline StyleGAN
    'config-b', # + Weight demodulation
    'config-c', # + Lazy regularization
    'config-d', # + Path length regularization
    'config-e', # + No growing, new G & D arch.
    'config-f', # + Large networks (default)

    # Table 2
    'config-e-Gorig-Dorig',   'config-e-Gorig-Dresnet',   'config-e-Gorig-Dskip',
    'config-e-Gresnet-Dorig', 'config-e-Gresnet-Dresnet', 'config-e-Gresnet-Dskip',
    'config-e-Gskip-Dorig',   'config-e-Gskip-Dresnet',   'config-e-Gskip-Dskip',
]

#----------------------------------------------------------------------------

def run(dataset, data_dir, result_dir, config_id, num_gpus, total_kimg, gamma, mirror_augment, metrics, resume_pkl, watermark_size, decoupleL2_weight, latentsRecL2_weight, watermarkCls_weight, res_modulated_range):
    assert '-' in res_modulated_range
    res_modulated_min_log2 = int(np.log2(int(res_modulated_range[:res_modulated_range.index('-')])))
    res_modulated_max_log2 = int(np.log2(int(res_modulated_range[res_modulated_range.index('-')+1:])))
    train     = EasyDict(run_func_name='training.training_loop.training_loop') # Options for training loop.
    E         = EasyDict(func_name='training.networks_stylegan2.E_stylegan2_watermark', watermark_size=watermark_size)       # Options for generator network.
    G         = EasyDict(func_name='training.networks_stylegan2.G_main_watermark', watermark_size=watermark_size, res_modulated_min_log2=res_modulated_min_log2, res_modulated_max_log2=res_modulated_max_log2)       # Options for generator network.
    D         = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
    EG_opt    = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
    D_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
    EG_loss   = EasyDict(func_name='training.loss.EG_logistic_ns_pathreg', decoupleL2_weight=decoupleL2_weight, latentsRecL2_weight=latentsRecL2_weight, watermarkCls_weight=watermarkCls_weight)      # Options for generator loss.
    D_loss    = EasyDict(func_name='training.loss.D_logistic_r1')              # Options for discriminator loss.
    sched     = EasyDict()                                                     # Options for TrainingSchedule.
    grid      = EasyDict(size='1080p', layout='random')                           # Options for setup_snapshot_image_grid().
    sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000, 'gpu_options.allow_growth': False, 'graph_options.place_pruned_graph': True} # Options for tflib.init_tf().

    train.data_dir = data_dir
    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.resume_pkl = resume_pkl
    sched.E_lrate_base = sched.G_lrate_base = sched.D_lrate_base = 0.002
    sched.minibatch_size_base = 8 * num_gpus
    sched.minibatch_gpu_base = 8
    D_loss.gamma = 10
    metrics = [metric_defaults[x] for x in metrics]
    desc = 'stylegan2'

    desc += '-' + dataset
    dataset_args = EasyDict(tfrecord_dir=dataset)

    #assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    assert config_id in _valid_configs
    desc += '-' + config_id

    desc += '_watermark_size_%f' % watermark_size
    desc += '_decoupleL2_weight_%f' % decoupleL2_weight
    desc += '_latentsRecL2_weight_%f' % latentsRecL2_weight
    desc += '_watermarkCls_weight_%f' % watermarkCls_weight
    desc += '_res_modulated_range_%s' % res_modulated_range

    # Configs A-E: Shrink networks to match original StyleGAN.
    if config_id != 'config-f':
        E.fmap_base = G.fmap_base = D.fmap_base = 8 << 10

    # Config E: Set gamma to 100 and override G & D architecture.
    if config_id.startswith('config-e'):
        D_loss.gamma = 100
        if 'Dorig'   in config_id: E.architecture = 'orig'
        if 'Dskip'   in config_id: E.architecture = 'skip'
        if 'Dresnet' in config_id: E.architecture = 'resnet' # (default)
        if 'Gorig'   in config_id: G.architecture = 'orig'
        if 'Gskip'   in config_id: G.architecture = 'skip' # (default)
        if 'Gresnet' in config_id: G.architecture = 'resnet'
        if 'Dorig'   in config_id: D.architecture = 'orig'
        if 'Dskip'   in config_id: D.architecture = 'skip'
        if 'Dresnet' in config_id: D.architecture = 'resnet' # (default)

    # Configs A-D: Enable progressive growing and switch to networks that support it.
    if config_id in ['config-a', 'config-b', 'config-c', 'config-d']:
        sched.lod_initial_resolution = 8
        sched.E_lrate_base = sched.G_lrate_base = sched.D_lrate_base = 0.001
        sched.E_lrate_dict = sched.G_lrate_dict = sched.D_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        sched.minibatch_size_base = 32 * num_gpus # (default)
        sched.minibatch_size_dict = {8: 256*num_gpus, 16: 128*num_gpus, 32: 64*num_gpus, 64: 32*num_gpus}
        sched.minibatch_gpu_base = 32 # (default)
        sched.minibatch_gpu_dict = {8: 256, 16: 128, 32: 64, 64: 32}
        G.synthesis_func = 'G_synthesis_stylegan_revised'
        D.func_name = 'training.networks_stylegan2.D_stylegan'

    # Configs A-C: Disable path length regularization.
    if config_id in ['config-a', 'config-b', 'config-c']:
        G_loss = EasyDict(func_name='training.loss.G_logistic_ns')

    # Configs A-B: Disable lazy regularization.
    if config_id in ['config-a', 'config-b']:
        train.lazy_regularization = False

    # Config A: Switch to original StyleGAN networks.
    if config_id == 'config-a':
        G = EasyDict(func_name='training.networks_stylegan.G_style')
        D = EasyDict(func_name='training.networks_stylegan.D_basic')

    if gamma is not None:
        D_loss.gamma = gamma

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(E_args=E, G_args=G, D_args=D, EG_opt_args=EG_opt, D_opt_args=D_opt, EG_loss_args=EG_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = result_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

_examples = '''examples:

  # Train StyleGAN2 using the FFHQ dataset
  python %(prog)s --num-gpus=8 --data-dir=~/datasets --config=config-f --dataset=ffhq --mirror-augment=true

valid configs:

  ''' + ', '.join(_valid_configs) + '''

valid metrics:

  ''' + ', '.join(sorted([x for x in metric_defaults.keys()])) + '''

'''

def main():
    parser = argparse.ArgumentParser(
        description='Train StyleGAN2.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--config', help='Training config (default: %(default)s)', default='config-e-Gskip-Dresnet', dest='config_id', metavar='CONFIG')
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg', help='Training length in thousands of images (default: %(default)s)', metavar='KIMG', default=88000, type=int)
    parser.add_argument('--gamma', help='R1 regularization weight (default is config dependent)', default=None, type=float)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)', default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)', default='fid_watermark_accuracy_50k', type=_parse_comma_sep)
    parser.add_argument('--resume-pkl', help='Pre-trained network path" (default: %(default)s)', default=None, type=str)

    parser.add_argument('--watermark-size', help='Watermark dimensionality (default: %(default)s)', default=128, type=int)
    parser.add_argument('--decoupleL2-weight', help='Weight for watermark decouple L2 loss (default: %(default)s)', default=2.0, type=float)
    parser.add_argument('--latentsRecL2-weight', help='Weight for normal latent L2 reconstruction loss (default: %(default)s)', default=1.0, type=float)
    parser.add_argument('--watermarkCls-weight', help='Weight for onehot watermark classification loss (default: %(default)s)', default=2.0, type=float)

    parser.add_argument('--res-modulated-range', help='Working range for modulated_conv2d_layer() (default: %(default)s)', default='4-128', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print ('Error: dataset root directory does not exist.')
        sys.exit(1)

    if args.config_id not in _valid_configs:
        print ('Error: --config value must be one of: ', ', '.join(_valid_configs))
        sys.exit(1)

    for metric in args.metrics:
        if metric not in metric_defaults:
            print ('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    run(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

