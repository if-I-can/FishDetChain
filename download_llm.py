"""
大模型下载脚本
"""

# import os
# import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
model_dir = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/home/zsl/home', revision='master')


# 大模型地址： 



huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /home/zsl/LLaMA-Factory/LLM