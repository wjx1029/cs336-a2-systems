# 1.Profiling and Benchmarking
## 1.1 Profiling
- Problem (benchmarking_script):  Benchmarking Script

  - small model
    - forward-only: mean=0.07 s, std=0.00026

    - forward-backward: mean=0.22 s, std=0.00030

    - forward-optimizer: mean=0.25 s, std=0.00051
 
   - medium model
      -  forward-only: mean=0.21 s, std=0.00061
      - forward-backward: mean=0.64 s, std=0.00085
      - forward-optimizer: mean=0.75 s, std=0.00080

- (c) 当warmup=0时,标准差会变大;预热会让系统趋于稳定, 避免第一次加载数据干扰测试
