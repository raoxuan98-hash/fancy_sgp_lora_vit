CUDA_VISIBLE_DEVICES=0 python3 -u main.py --dataset "imagenet-r" --smart_defaults --distillation_transform 'linear'&

sleep 15

CUDA_VISIBLE_DEVICES=1 python3 -u main.py --dataset "cub200_224" --smart_defaults --distillation_transform 'linear'& 

sleep 15

CUDA_VISIBLE_DEVICES=2 python3 -u main.py --dataset "imagenet-r" --smart_defaults --distillation_transform 'identity' &

sleep 15

CUDA_VISIBLE_DEVICES=4 python3 -u main.py --dataset "cub200_224" --smart_defaults --distillation_transform 'identity' & 