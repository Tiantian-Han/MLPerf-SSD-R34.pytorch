# MLPerf_SSD-R34-Large
SSD on Large images with a backbone of ResNet34 based on MLPerf-training single-stage-detector repo 
##Installation
To install the environment please follow the instruction on [MLPerf-training single-stage-detector](https://github.com/mlperf/training/tree/master/single_stage_detector). The files in this repo replace the files in the ssd folder.

## Changes from original repo:
1. Support training on any data size including images with uneven ratio e.g 1600x1200
2. Support training on multi GPUs
3. Support different strides from command line - this is a list of 6 numbers: default [1,1,2,2,2,1]. The idea is to control the number of anchors 
3. Removed hard coded steps\feature maps sizes 

## Experiments (TO DO):
1. Train 1400x1400 images on 8 GPUs with batch of 32 with the diffult anchors 
   ```
   python train.py --device-ids 0 1 2 3 4 5 6 7
   ```
2. Train of 1400x1400 with smaller feature maps size by changing the strides - this would results with less anchors.
   ```
   python train.py --device-ids 0 1 2 3 4 5 6 7 --strides 4 4 2 2 2 1
   ```
3. Train on images of size 1600x1200 (asspect ratio would need to be adjusted - line 101 in train.py)
   ```
   python train.py --device-ids 0 1 2 3 4 5 6 7 --image_size 1600 1200
   ```
   
  
To resume training please use the ```--checkpoint``` flag so you can load your pre-trained 300x300 models and check results on a larger image size. 
