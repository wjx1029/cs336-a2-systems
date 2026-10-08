import torch
import timeit
import argparse
import numpy as np
import math

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
import cs336_basics.model

import torch.cuda.nvtx as nvtx

# 1. 使用装饰器给整个函数打一个最外层的标签
@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
   
    # 2. 内部用 with 语句给不同阶段打上子标签
    with nvtx.range("computing attention scores"):
        d_k = K.shape[-1]
        # 计算 Q 和 K 的注意力分数 (例如: scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    
    with nvtx.range("applying mask"):
        if mask is not None:
            scores = torch.where(mask, scores, float("-inf"))
        
    with nvtx.range("computing softmax"):
        # 计算 softmax (例如: attn = torch.softmax(scores, dim=-1))
        attn = torch.softmax(scores, dim=-1)
        
    with nvtx.range("final matmul"):
        # 计算输出投影 (例如: output = attn @ V)
        output = attn @ V
        
    return output


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='高级示例')

    # 模型超参数
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)

    # 测试参数
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--m_steps', type=int, default=10)
    parser.add_argument('--mode', choices=['forward-only', 'forward-backward', 'forward-optimizer'], default='forward-optimizer')

    return parser.parse_args()


def benchmark():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires an NVIDIA GPU with CUDA.")

    args = parse_args()

    device = torch.device('cuda:0')

    model = BasicsTransformerLM(vocab_size=args.vocab_size,
                                context_length=args.context_length,
                                d_model=args.d_model,
                                d_ff=args.d_ff,
                                num_heads=args.num_heads,
                                num_layers=args.num_layers).to(device)

    # 随机生成batch数据
    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).to(device)

    # forward-only
    if args.mode == 'forward-only':
        # warmup
        for _ in range(args.warmup):
            logits = model(inputs)
        torch.cuda.synchronize()

        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        
        # measure
        logs = []
        for _ in range(args.m_steps):
            start = timeit.default_timer()
            logits = model(inputs)
            torch.cuda.synchronize()
            end = timeit.default_timer()
            logs.append(end - start)

    # forward-backward
    if args.mode == 'forward-backward':
        optimizer = AdamW(model.parameters())

        # warmup
        for _ in range(args.warmup):
            optimizer.zero_grad()
            loss = model(inputs).sum()
            loss.backward()
        torch.cuda.synchronize()

        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

        # measure
        logs = []
        for _ in range(args.m_steps):
            optimizer.zero_grad()
            start = timeit.default_timer()
            loss = model(inputs).sum()
            loss.backward()
            torch.cuda.synchronize()
            end = timeit.default_timer()
            logs.append(end - start)           

    # forward-optimizer
    if args.mode == 'forward-optimizer':
        optimizer = AdamW(model.parameters())

        # warmup
        for _ in range(args.warmup):
            optimizer.zero_grad()
            loss = model(inputs).sum()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

        # measure
        logs = []
        for _ in range(args.m_steps):
            optimizer.zero_grad()
            start = timeit.default_timer()
            loss = model(inputs).sum()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            end = timeit.default_timer()
            logs.append(end - start) 

    logs = np.array(logs)
    print(f'{args.mode}: mean={np.mean(logs):.2f} s, std={np.std(logs):.5f}')

if __name__ == "__main__":

    benchmark()

