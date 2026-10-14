# 1.Profiling and Benchmarking
## 1.1 Benchmarking
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

## 1.2 Profiling
- (a) forward pass spend about 1 second, higher than standrad python library.
- (b) **volta_sgemm_128x128_tn** and **volta_sgemm_128x64_tn** take the most cumulative GPU  time during forward pass.  They are invoked during a single forward pass in 375 and 900 times. The most runtime taked up by CUDA kernal are not same while do both forward and backward.
- (c) vectorized_elementwise_kernel, elementwise_kernel, softmax_warp_forward, CatArrayBatchedCopy, reduce_kernel
- (d) he fraction of time spent on matrix multiplication decreases in a full training step compared to inference-only, because backpropagation and the AdamW optimizer step add substantial non-matmul work (elementwise and reduction kernels), which increases the share of "other kernels."
- (e) The FLOPs differ by ~300×, yet the runtimes are essentially the same, because softmax is memory-bound while matrix multiplication is compute-bound.

## 1.3 Mixed Precision
1. The first method is almost right, the second precision is lower, the third and the fourth precision better than second ,but lower than the first.
2. - the model parameters within the autocast context?  FP32
   - the output of the first feed-forward layer (ToyModel.fc1)? FP16
   - the output of layer norm (ToyModel.ln)?  FP32
   - the model’s predicted logits?  FP16
   - the loss?  FP32
   - the model’s gradients? FP32
3. Layer normalization is sensitive to mixed precision because it involves reductions over many elements (computing the mean and variance) followed by subtraction of the mean and division by the standard deviation. In FP16, these operations suffer from overflow/underflow and catastrophic cancellation, since the variance may be very small while individual values are large, and the dynamic range of FP16 (≈ 6×10⁴) is easily exceeded when squaring and summing. With BF16, we still generally keep LayerNorm in higher precision, because although BF16 has the same exponent range as FP32 (avoiding overflow), it has only 8 bits of mantissa, so accumulating many values and computing small variances still loses substantial precision—hence both FP16 and BF16 typically keep LayerNorm (and softmax/loss) in FP32.
4. Mixed precision consistently reduces both forward and backward time. As model size grows, the speedup increases and gradually saturates. BF16 is slower FP16, but BF16 is numerically safer (no GradScaler needed, wider exponent range) and is therefore often preferred in practice.

## 1.4 Profiling Memory
(a) 
