# MLPerf Inference SSD Large
Two-scale SSD Inference with pretrained model based on MLPerf-training single-stage-detector repo.

# Installation
To install the environment please follow the instruction on [MLPerf-training single-stage-detector](https://github.com/mlperf/training/tree/master/single_stage_detector). The files in this repo replace the files in the ssd folder.

## Models
Pretrained models can be found in [google drive](https://drive.google.com/open?id=1Gf3hdIdmZUwrPP4mLvNu1CK-ZNqXb38z).

## Parameters
- image-size: two image sizes for two-scale inference (e.g., image size model was trained on, double that value)
- dboxes-scale: multiplicative factor for default box scales in mlperf reference (i.e, dboxes-scale * [21, 45, 99, 153, 207, 261, 315])
- dboxes-steps: default boxes steps (e.g., mlperf reference uses [8, 16, 32, 64, 100, 300])
- small-mask: binary mask to select feature maps for small-scale inference
- large-mask: binary mask to select feature maps for large-scale inference
- use-filter: filter small/medium detections for small-scale inference and big detections for large-scale inference

## How to run
1. Two-scale inference on 300 x 300 and 600 x 600 images using model trained on 300 x 300 images:
   ```
   python inference.py --data /coco --image-size 300 300 --dboxes-scale 1 --model model300.pt --use-fp16 --dboxes-steps 8 16 32 64 100 300
   ```

2. Two-scale inference on 700 x 700 and 1400 x 1400 images using model trained on 700 x 700 images:
   ```
   python inference.py --data /coco --image-size 700 1400 --dboxes-scale 2 --model model700.pt --use-fp16
   ```

3. Same as 2 but removing detections with given area:
   ```
    python inference.py --data /coco --image-size 700 1400 --dboxes-scale 2 --model model700.pt --use-fp16 --use-filter
   ```

4. Same as 2 but removing feature maps:
   ```
    python inference.py --data /coco --image-size 700 1400 --dboxes-scale 2 --model model700.pt --use-fp16 --small-mask 0 1 1 1 1 1 --large-mask 1 1 1 1 0 0
   ```

