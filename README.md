<h1 align="center">Code for NeurIPS Submission</h1>


<h4 align="center">ActMAD: Activation Matching to Align Distributions for Test-Time-Training</h4>

This repository contains all the code for reproducing the results listed in our paper. Please follow
the steps below to run the experiments. 

# 1. Analyzing Results 
To only access the detailed evaluation results, run the following commands.
(For this, you need a ```python3``` environment with ```numpy``` and ```matplotlib``` installed):

- To show the detailed results for a specific table from the paper, e.g. Table 2, run:
    
      python results_analysis.py --table 2

- To show detailed ablation results and plot all figures, run:  

      python results_analysis.py --abl_only --plot 

- To show results for all tables, ablation studies and plot all accompanying figures, run: 

      python results_analysis.py --all --plot 

Results for all random (10) runs, ablation studies and further additional results are provided in the ```results``` folder 
as ```numpy arrays```.

# 2. Install Requirements
        pip install -r requirements.txt

# 3. Preparing Datasets
* Download the original train and test set for [ImageNet](https://image-net.org/download.php) & [ImageNet-C](https://zenodo.org/record/2235448#.Yn5OTrozZhE) datasets.
* Download the original train and test set for [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) & [CIFAR-10C](https://zenodo.org/record/2535967#.Yn5QwbozZhE) & [100C](https://zenodo.org/record/3555552#.Yn5QwLozZhE) datasets.
* Download Clear (Original) [KITTI dataset](http://www.cvlibs.net/datasets/kitti/).
* Download [KITTI-Fog/Rain](https://team.inria.fr/rits/computer-vision/weather-augment/) datasets.
* Super-impose snow on KITTI dataset through this [repository](https://github.com/hendrycks/robustness).

Please place the original Imagnet and ImageNet-C datasets in a single folder. Same must be done
for CIFAR-10/100 and CIFAR-10/100C datasets.

KITTI datasets should be divided and placed in 4 separate folders (one for each weather condition).
The ```train & validation``` splits used to obtain results in the paper can be found in ```./data/splits```.

In each folder the structure should be as following example:
```
KITTI_Fog_Dataset
│   images
│   labels   
│   train.txt
|   val.txt
```
We recommend using our provided splits in ```./data/splits``` as the checkpoint is trained on these splits. However, you can create your own splits and train
another model by using the [PyTorch implementation](https://github.com/ultralytics/yolov3) for Yolov3.

# 4. Pre-Trained Models

* ResNet-18/50: ImageNet pre-trained models are automatically downloaded from PyTorch. 
* DeepAugment: Please download [DeepAug](https://arxiv.org/abs/2006.16241) 
  pre-trained ResNet-50 from 
  [here](https://drive.google.com/file/d/1Uy8g-yImIzBs2b4Ry2b3gMGFnEnxc4yx/view?usp=sharing) 
  and place them in ```./ckpt/```.
* CIFAR-10/100C: Pre-Trained ResNet-26 and 
  [AugMix](https://github.com/google-research/augmix) pre-trained WRN-40-2 are already 
  in ```./ckpt/```.
* Download KITTI pre-trained [Yolov3](https://arxiv.org/abs/1804.02767) from [here](https://drive.google.com/file/d/1NWwhX7zmsQh0791VUL_5mB9cF29Xd0SU/view?usp=sharing) and place in the same directory as above.




# 5. Running Experiments 

All experiments will be started by using a single command.

        sh run_actmad.sh

Please make sure you change the paths to the dataset in the shell scrip before you start. 
By default, for ImageNet-C the experiments will start by using ```batchsize=10``` and only 
```1%``` of train and test data. 
Results will be saved in the ```./results/``` folder upon completion of each experiment. 

If you wish to run individual experiments you can simply run the run following commands.

### ImageNet-C
Before starting the ImageNet-C experiments, you will need to save the activation statistics
from the training data. You can either save them for all the ResNet backbones we have 
used in the paper with the following command (this will automatically save it for all data 
fractions): 
        
        python save_stats_inc.py

Or for quick experiments you can download the statistics we have already computed for 
different data fractions for [DeepAug](https://arxiv.org/abs/2006.16241) pre-trained 
ResNet-50 model from [here](https://drive.google.com/drive/folders/1Jvijjfj0aZ41jcm5hfYHqWElgN_IR92F?usp=sharing) 
and place them in the main folder. After download simply run: 

    python main_inc.py --dataroot PATH_TO_DATA_FOLDER --models [deep_aug]

### Cifar-10/100C 
These experiments should run without downloading anything because statistics are computed
on the fly as the dataset is extremely light weight. Simply run the following commands and 
results will be saved in the ```results``` folder for all backbones automatically. 

    python main_c10.py --dataroot PATH_TO_DATA_FOLDER

    python main_c100.py --dataroot PATH_TO_DATA_FOLDER

### KITTI-FOG/RAIN/SNOW
After downloading the datasets, checkpoint and changing the paths in the configurations (see instructions 
in ```./data/weather_data/instructions.txt```). Individual KITTI experiments can be started with 
the following commands: 

    python main_kitti.py --weights ckpt/clear_kitti.pt --data data/weather_data/kitti_fog.yaml
    
    python main_kitti.py --weights ckpt/clear_kitti.pt --data data/weather_data/kitti_rain.yaml

    python main_kitti.py --weights ckpt/clear_kitti.pt --data data/weather_data/kitti_snow.yaml
