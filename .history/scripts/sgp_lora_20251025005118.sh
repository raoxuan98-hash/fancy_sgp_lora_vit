

CUDA_VISIBLE_DEVICES=0 python3 -u main.py --dataset "cars196_224" --smart_defaults &

sleep 15

CUDA_VISIBLE_DEVICES=1 python3 -u main.py --dataset "cifar100_224" --smart_defaults &

sleep 15


CUDA_VISIBLE_DEVICES=2 python3 -u main.py --dataset "imagenet-r" --smart_defaults &

sleep 15

CUDA_VISIBLE_DEVICES=3 python3 -u main.py --dataset "cub200_224" --smart_defaults --weight_temp 1.0&

wait

CUDA_VISIBLE_DEVICES=0 python3 -u main.py --dataset "cars196_224" --smart_defaults --weight_temp 1.0&

sleep 15

CUDA_VISIBLE_DEVICES=1 python3 -u main.py --dataset "cifar100_224" --smart_defaults &

sleep 15


CUDA_VISIBLE_DEVICES=2 python3 -u main.py --dataset "imagenet-r" --smart_defaults &

sleep 15

CUDA_VISIBLE_DEVICES=3 python3 -u main.py --dataset "cub200_224" --smart_defaults &