import os
from huggingface_hub import snapshot_download
import logging

# 1. 设置 Hugging Face 镜像（国内加速）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 设置日志级别（确保进度条可见）
logging.basicConfig(level=logging.INFO)

# 3. 下载模型并显示进度条
snapshot_download(
    repo_id="timm/vit_base_patch16_224.mae",  # 模型名称
    local_dir="./pretrained_models",          # 保存路径
    resume_download=True,                     # 断点续传
    local_dir_use_symlinks=False,             # 避免符号链接（可选）
    tqdm=True                                # 显示进度条
)