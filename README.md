# dawnbench_inference_imagenet

## run inference
```
taskset -c 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 python inference.py \
--model resnet50_imagenet_int8.pb --data_dir /ramdisk/ --batch_size 1 --intra 1 --inter 16
```
