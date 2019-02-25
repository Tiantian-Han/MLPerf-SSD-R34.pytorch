export TORCH_MODEL_ZOO=/mlperf
python inference.py --data /coco --use-fp16 \
    --image-size 700 1400 --dboxes-scale 2 --model model700.pt \
    --small-mask 0 1 1 1 1 1 \
    --large-mask 1 1 1 1 0 0 \
    #--dboxes-steps 8 16 32 64 100 300 # mlperf steps
