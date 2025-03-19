"""
大模型下载脚本
"""

# import os
# import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
model_dir = snapshot_download('Qwen/Qwen2.5-7B', cache_dir='/home/zsl/FishDetChain', revision='master')


# 大模型地址： 



# # huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /home/zsl/FishDetChain
# CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2_5-7B --tokenizer Qwen/Qwen2_5-7B --port 11433 --api-key ollama