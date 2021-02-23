# Scalable GAN Fingerprints

### [Responsible Disclosure of Generative Models Using Scalable Fingerprinting](https://arxiv.org/pdf/2012.08726)
[Ning Yu](https://sites.google.com/site/ningy1991/)\*, Vladislav Skripniuk\*, [Dingfan Chen](https://cispa.de/en/people/dingfan.chen), [Larry Davis](http://users.umiacs.umd.edu/~lsd/), [Mario Fritz](https://cispa.saarland/group/fritz/)<br>
arXiv 2021

<img src='fig/teaser.png' width=600>

## Abstract
Over the past six years, deep generative models have achieved a qualitatively new level of performance. Generated data has become difficult, if not impossible, to be distinguished from real data. While there are plenty of use cases that benefit from this technology, there are also strong concerns on how this new technology can be misused to spoof sensors, generate deep fakes, and enable misinformation at scale. Unfortunately, current deep fake detection methods are not sustainable, as the gap between real and fake continues to close. In contrast, our work enables a responsible disclosure of such state-of-the-art generative models, that allows researchers and companies to fingerprint their models, so that the generated samples containing a fingerprint can be accurately detected and attributed to a source. Our technique achieves this by an efficient and scalable ad-hoc generation of a large population of models with distinct fingerprints. Our recommended operation point uses a 128-bit fingerprint which in principle results in more than 10<sup>36</sup> identifiable models. Experiments show that our method fulfills key properties of a fingerprinting mechanism and achieves effectiveness in deep fake detection and attribution.

## Prerequisites
- Linux
- NVIDIA GPU + CUDA 10.0 + CuDNN 7.5
- Python 3.6
- tensorflow-gpu 1.14
- To install the other Python dependencies, run `pip3 install -r requirements.txt`.

## Datasets
We experiment on three datasets:
- [CelebA](https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8). We use the first 30k images and crop them centered at (x,y) = (89,121) with size 128x128. To prepare the dataset, first download and unzip the aligned png images to `celeba/Img/`, then run
  ```
  python3 dataset_tool.py create_celeba \
  datasets/celeba_align_png_cropped_30k \
  celeba/Img/img_align_celeba_png \
  --num_images 30000
  ```
  where `datasets/celeba_align_png_cropped_30k` is the output directory containing the prepared data format that enables efficient streaming for our training, and `celeba/Img/img_align_celeba_png` is the input directory containing CelebA png files.
  
- [LSUN Bedroom](https://github.com/fyu/lsun). Similarly, we use the first 30k training images and resize them to 128x128. To prepare the dataset, first download and extract the png images to `lsun_bedroom_train`, then run
  ```
  python3 dataset_tool.py create_from_images \
  datasets/lsun_bedroom_train_30k_128x128 \
  lsun_bedroom_train \
  --shuffle 0 \
  --num_images 30000 \
  --resolution 128
  ```

- [LSUN Cat](http://dl.yf.io/lsun/objects/). Similarly, we use the first 50k images at the original 256x256 size. To prepare the dataset, first download and extract the png images to `lsun_cat`, then run
  ```
  python3 dataset_tool.py create_from_images \
  datasets/lsun_cat_50k_256x256 \
  lsun_cat \
  --shuffle 0 \
  --num_images 50000 \
  --resolution 256
  ```

## Training
- Run, e.g.,
  ```
  python3 run_training.py --data-dir=datasets --config=config-e-Gskip-Dresnet --num-gpus=2 \
  --dataset=celeba_align_png_cropped_30k \
  --result-dir=results/celeba_align_png_cropped_30k \
  --watermark-size=128 \
  --res-modulated-range=4-128 \
  --metrics=fid_watermark_accuracy_30k
  ```
  where
  - `result-dir` contains model snapshots `network-snapshot-*.pkl`, real samples `reals.png`, randomly generated samples `fakes-arbitrary-*.png` at different snapshots, randomly generated samples `fakes-watermarks-same-*.png` at different snapshots with arbitrary latent code and the same watermark (fingerprint), randomly generated samples `fakes-latents-same-*.png` at different snapshots with the same latent code and arbitrary watermarks (fingerprints), log file `log.txt`, tensorboard plots `events.out.tfevents.*`, and so on.
  - `watermark-size`: The number of bits of embedded watermark (fingerprint).
  - `res-modulated-range`: At which resolutions of generator layers to modulate watermark (fingerprint). **Our experiments show modulating at all resolutions achieves the optimal performance in general.**
  - `metrics`: Evaluation metric(s). `fid_watermark_accuracy_30k` measures (1) the Fréchet inception distance between 30k randomly generated samples and 30k real samples, and (2) the bitwise accuracy of watermark (fingerprint) detection. The evaluation result is save in `results/metric-fid_watermark_accuracy_30k.txt`.

## Pre-trained models
- The pre-trained scalable GAN fingerprinting models can be downloaded from:
  - [30k CelebA 128x128](https://drive.google.com/file/d/1ODFds30TIGO-qRl1vbMVsxvivP1loHOz/view?usp=sharing)
  - [30k LSUN Bedroom 128x128](https://drive.google.com/file/d/1legqK4V9nUrY1_m0iKG1-fMGg6Rixjfn/view?usp=sharing)
  - [50k LSUN Cat 256x256](https://drive.google.com/file/d/1I9ZFVwK6OKBY945flRaLnf7CAD8Q0Ejo/view?usp=sharing)
