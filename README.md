# Flash Attention: Fast and Memory-Efficient Attention Mechanism

## Overview
Flash Attention is a fast, memory-efficient, and IO-aware implementation of attention mechanisms used in deep learning models. 
This project showcases a simplified version of Flash Attention on GPUs using CUDA with randomly initialized matrix inputs, demonstrating how significant performance improvements can be achieved with optimized GPU compute, over traditional attention implementations.

## Key Features
1. **Fast**: Speeds up training for large models like BERT and GPT by up to 3x.
2. **Memory-Efficient**: Reduces memory complexity from \(O(N^2)\) to \(O(N)\) by leveraging hardware memory hierarchies.
3. **Exact**: Ensures accuracy without approximation.
4. **IO-Aware**: Optimizes memory access and communication for modern GPUs.

## Implementation Highlights
Flash Attention is implemented in three versions:
1. **CPU Implementation**: Provides a baseline implementation using sequential processing.
2. **Naive GPU Implementation**: Uses CUDA global memory for parallel computations.
3. **Optimized GPU Implementation**: Leverages shared memory, memory tiling, thread coarsening, and memory coalescing for maximum efficiency.

## Key Optimizations
1. **Memory Tiling**: Divides computation into manageable tiles to enhance memory locality.
2. **Thread Coarsening**: Increases the work assigned per thread to reduce overhead.
3. **Shared Memory**: Uses fast on-chip shared memory to minimize global memory access.
4. **Memory Coalescing**: Ensures efficient memory access patterns for threads.

## Experimental Results
Performance benchmarks were conducted on an NVIDIA GeForce RTX 4050 Laptop GPU with varying configurations. Below are the latency results for 1024 x 1024 matrices:

| Implementation            | Latency    |
|---------------------------|------------|
| CPU                       | 9,427 ms   |
| GPU (Naive)               | 38.78 ms   |
| GPU (Optimized Shared)    | 14.12 ms   |

The optimized implementation achieves a **3x speedup** compared to the naive GPU version and a **600x improvement** over the CPU baseline.

## Source Code Structure
- **CPU Implementation**: Implements scaled dot-product attention using sequential processing.
- **Naive GPU Implementation**: Implements CUDA kernels using global memory for QK^T, softmax, and output computation.
- **Optimized GPU Implementation**: Implements CUDA kernels using shared memory, tiling, and other optimizations for improved performance.

## How to Build and Run
1. Clone this repository:
   ```bash
   git clone https://github.com/damienjose/cuda-flashattention
   cd cuda-flashattention
   ```

2. Set up your CUDA environment:
   - Install the latest CUDA toolkit.
   - Update the `sm` and `compute` settings in the Visual Studio project properties, specific for your GPU. See https://www.truehost.com/what-is-compute-capability-of-a-gpu/

3. Build the project:
   ```bash
   nvcc -o cuda-flashattention kernel.cu -arch=sm_80
   ```

4. Run the program:
   ```bash
   ./cuda-flashattention
   ```

## Profiling and Performance Analysis
- The naive GPU implementation is memory-bound due to frequent global memory accesses.
- The optimized shared memory implementation transforms memory-bound operations into compute-bound ones.
- Profiling tools like NVIDIA Nsight can be used for further performance analysis.

## References
1. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
2. [Understanding Flash-Attention and Flash-Attention-2](https://towardsai.net/p/artificial-intelligence/understanding-flash-attention-and-flash-attention-2-the-path-to-scale-the-context-lenght-of-language-models)
3. [GitHub - Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

## Authors
- Bryan Zhou ([bryanzhou2000@gmail.com](mailto:bryanzhou2000@gmail.com))
- Damien Jose ([damien.jose@gmail.com](mailto:damien.jose@gmail.com))

## Acknowledgments
This project was done as part of the AUT24 GPU Compute course at the University of Washington. We thank Professor Colin N. Reinhardt and TAs Arnab Karmakar, Xiaxi Shen for guiding us throughout this course. We also appreciate the authors of Flash Attention for their contributions and open sourcing their code.
