#! /usr/bin/env bash


export PYTHONPATH=$PYTHONPATH:$(pwd)

# ===================================
models_inc='deep_aug res_50 res_18' # for imagenet exp

# please update the folder locations where your datasets are placed
dataroot_c10_c100=PLEASE_CHANGE_ME
dataroot_inc=PLEASE_CHANGE_ME

# please update the paths to config files for different kitti experiments (examples in ./data/weather_data/)
kitti_fog_config=PLEASE_CHANGE_ME
kitti_rain_config=PLEASE_CHANGE_ME
kitti_snow_config=PLEASE_CHANGE_ME

# please make sure you have downloaded the kitti clean weights
kitti_weights=./ckpt/clear_kitti.pt

echo 'DATASET PATH C10/100: '${dataroot_c10_c100}
echo 'DATASET PATH ImageNet-C: '${dataroot_inc}
echo 'Models fo ImageNet-C: '${models_inc}

# ===================================

printf 'Saving ImageNet Statistics for all backbones...'

python save_stats_inc.py

printf 'Starting ImageNet-C Adaptation for different Dataset Sizes and Batch Sizes...'

python main_inc.py --dataroot ${dataroot_inc} --models ${models_inc}

printf 'Starting CIFAR-10C experiments for different backbones...'

python main_c10.py --dataroot ${dataroot_c10_c100}

printf 'Starting CIFAR-100C experiments for different backbones...'

python main_c100.py --dataroot ${dataroot_c10_c100}

printf 'Starting KITTI-Fog Experiment...'

python main_kitti.py --adapt True --data ${kitti_fog_config} --weights ${kitti_weights}

printf 'Starting KITTI-Rain Experiment...'

python main_kitti.py --adapt True --data ${kitti_rain_config} --weights ${kitti_weights}

printf 'Starting KITTI-Snow Experiment...'

python main_kitti.py --adapt True --data ${kitti_snow_config} --weights ${kitti_weights}
