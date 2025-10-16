CUDA_VISIBLE_DEVICES=5 python3 -u main.py --dataset "imagenet-r" --smart_defaults &

sleep 120

CUDA_VISIBLE_DEVICES=1 python3 -u main.py --dataset "cifar100_224" --smart_defaults &

sleep 120

CUDA_VISIBLE_DEVICES=2 python3 -u main.py --dataset "cars196_224" --smart_defaults &

sleep 120

CUDA_VISIBLE_DEVICES=4 python3 -u main.py --dataset "cub200_224" --smart_defaults & 