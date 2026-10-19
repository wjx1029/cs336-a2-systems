# Memory Profile
## 1
run 

```
uv run benchmark.py --mode=forward-only --m-steps=1 --memory-profile=true
uv run benchmark.py --mode=forward-optimizer --m-steps=1 --memory-profile=true
```

answer(a)

## 2
run

```
uv run benchmark.py --mode=forward-only --m-steps=1 --memory-profile=true --context_length=128
uv run benchmark.py --mode=forward-only --m-steps=1 --memory-profile=true --context_length=1024
uv run benchmark.py --mode=forward-optimizer --m-steps=1 --memory-profile=true --context_length=128
uv run benchmark.py --mode=forward-optimizer --m-steps=1 --memory-profile=true --context_length=1024
```

answer(b)

## 3
run

```
uv run benchmark.py --mode=forward-only --m-steps=1 --memory-profile=true --mixed_precision=float16
uv run benchmark.py --mode=forward-only --m-steps=1 --memory-profile=true --mixed_precision=bfloat16
uv run benchmark.py --mode=forward-optimizer --m-steps=1 --memory-profile=true --mixed_precision=float16
uv run benchmark.py --mode=forward-optimizer --m-steps=1 --memory-profile=true --mixed_precision=bfloat16
```

answer(c)

## 4

B * S * D

answer(d)

## 5

look closely at the “Active Memory Timeline” from `memory_snapshot_forward-only_512_none.pikle`

answer(e)

## 6

```
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --pytorch=autograd-shapes-nvtx \
    --cuda-memory-usage=true \
    -o transformer_profile \
    uv run benchmark.py --m_steps=1 --mode=forward-backward
```

answer(f)