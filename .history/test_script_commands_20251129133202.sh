#!/bin/bash

# 验证脚本：测试命令生成
echo "测试1: 使用默认参数"
echo "命令: ./sh/run_within_domain_evaluation_configurable.sh"
echo ""

echo "测试2: 指定架构和LoRA类型"
echo "命令: ./sh/run_within_domain_evaluation_configurable.sh vit-b-p16 basic_lora"
echo ""

echo "测试3: 不同的架构"
echo "命令: ./sh/run_within_domain_evaluation_configurable.sh vit-b-p16-dino sgp_lora"
echo ""

echo "测试4: 无效的架构（应该报错）"
echo "命令: ./sh/run_within_domain_evaluation_configurable.sh invalid_arch basic_lora"
echo ""

echo "生成示例命令（不执行）:"
dataset="imagenet-r"
gpu_id="0"
vit_type="vit-b-p16"
lora_type="basic_lora"
seed="1993"

cmd="python3 main.py \
    --dataset $dataset \
    --vit_type $vit_type \
    --lora_type $lora_type \
    --seed_list $seed \
    --smart_defaults \
    --cross_domain False"

echo "示例执行命令:"
echo "$cmd"
echo ""

echo "支持的参数："
echo "架构类型: vit-b-p16, vit-b-p16-dino, vit-b-p16-mae, vit-b-p16-clip, vit-b-p16-mocov3"
echo "LoRA类型: basic_lora, sgp_lora, nsp_lora, full"
echo ""
echo "使用方法:"
echo "./sh/run_within_domain_evaluation_configurable.sh [vit_type] [lora_type]"