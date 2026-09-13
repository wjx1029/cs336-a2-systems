import torch
import timeit
import argparse
import numpy as np

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='高级示例')

    # 模型超参数
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--d_ff', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--num_heads', type=int, required=True)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)

    # 测试参数
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--m_steps', type=int, default=10)
    parser.add_argument('--mode', choices=['forward-only', 'forward-backward', 'forward-optimizer'], default='forward-only')

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
    print(f'{args.mode}: mean={np.mean(logs):.2f}s, std={np.std(logs):.2f}s')

if __name__ == "__main__":

    benchmark()

