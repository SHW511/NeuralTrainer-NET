# Transformer Migration - Task Breakdown

This document contains actionable tasks for each milestone. Tasks are ordered by dependency and priority.

**Last Updated:** December 2024

## Completed Tasks Summary
- [x] Phase 1: Foundation Components (CPU implementations)
- [x] Phase 2: Attention Mechanism (CPU implementations)
- [x] Phase 3: Transformer Building Blocks (CPU implementations)
- [x] Phase 4: Complete Transformer Model (CPU implementations)
- [ ] Phase 5: Enhanced Text Processing (pending)
- [ ] Phase 6: Training Infrastructure (pending)
- [ ] Phase 7: Inference Optimization (pending)
- [ ] CUDA Acceleration for new components (pending)

---

## Phase 1: Foundation Components

### 1.1 Tensor Operations Library

#### Task 1.1.1: Create Tensor Class ✅ COMPLETED
**Priority:** Critical | **Complexity:** Medium | **Status:** DONE

**Description:** Create a `Tensor` class that properly handles multi-dimensional arrays with shape tracking.

**Acceptance Criteria:**
- [x] Support arbitrary dimensions (1D, 2D, 3D, 4D)
- [x] Track shape as `int[]`
- [x] Support reshape operations
- [x] Support view/slice operations
- [x] Implement proper memory layout (row-major)

**Files:**
- `NeuralNetwork/Tensors/Tensor.cs` ✅

**Code Skeleton:**
```csharp
public class Tensor
{
    public float[] Data { get; }
    public int[] Shape { get; }
    public int Rank => Shape.Length;
    public int Size => Data.Length;

    public Tensor(int[] shape);
    public Tensor(float[] data, int[] shape);
    public Tensor Reshape(int[] newShape);
    public Tensor Transpose(int dim1, int dim2);
    public float this[params int[] indices] { get; set; }
}
```

---

#### Task 1.1.2: Implement Batched Matrix Multiplication (CPU) ✅ COMPLETED
**Priority:** Critical | **Complexity:** Medium | **Status:** DONE

**Description:** Implement batched matrix multiplication for 3D tensors.

**Acceptance Criteria:**
- [x] Handle [B, M, K] × [B, K, N] → [B, M, N]
- [x] Handle broadcasting when batch dim is 1
- [x] Validate shape compatibility
- [ ] Unit tests with known values

**Files:**
- `NeuralNetwork/Tensors/TensorOperations.cs` ✅

**Algorithm:**
```
For each batch b:
    C[b] = MatMul(A[b], B[b])
```

---

#### Task 1.1.3: Implement Batched Matrix Multiplication (CUDA)
**Priority:** Critical | **Complexity:** High

**Description:** Create CUDA kernel for batched matrix multiplication.

**Acceptance Criteria:**
- [ ] Efficient parallel implementation
- [ ] Proper block/thread sizing
- [ ] Handle arbitrary batch sizes
- [ ] Match CPU output within epsilon

**Files:**
- `NeuralNetwork/Tensors/TensorOperationsCuda.cs`
- `NeuralNetwork/CU/TensorKernels.cu`

**CUDA Kernel Structure:**
```cuda
__global__ void BatchedMatMul(
    float* A, float* B, float* C,
    int batchSize, int M, int K, int N)
{
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // ... implementation
}
```

---

#### Task 1.1.4: Implement Transpose Operations ✅ COMPLETED
**Priority:** Critical | **Complexity:** Medium | **Status:** DONE

**Description:** Implement tensor transpose for arbitrary dimension swaps.

**Acceptance Criteria:**
- [ ] Swap any two dimensions
- [ ] Handle batched tensors correctly
- [ ] CPU and CUDA versions
- [ ] Validate output shapes

**Files:**
- `NeuralNetwork/Tensors/TensorOperations.cs`
- `NeuralNetwork/Tensors/TensorOperationsCuda.cs`

---

#### Task 1.1.5: Implement Element-wise Operations
**Priority:** High | **Complexity:** Low

**Description:** Add, multiply, scale tensors element-wise.

**Acceptance Criteria:**
- [ ] Add two tensors with broadcasting
- [ ] Multiply two tensors with broadcasting
- [ ] Scale tensor by scalar
- [ ] CUDA versions for all

**Files:**
- `NeuralNetwork/Tensors/TensorOperations.cs`

---

### 1.2 Layer Normalization

#### Task 1.2.1: Implement LayerNorm Forward Pass (CPU)
**Priority:** Critical | **Complexity:** Medium

**Description:** Implement layer normalization forward pass.

**Acceptance Criteria:**
- [ ] Compute mean along last dimension
- [ ] Compute variance along last dimension
- [ ] Apply normalization with learnable gamma/beta
- [ ] Handle epsilon for numerical stability

**Files:**
- `NeuralNetwork/Layers/LayerNorm.cs`

**Formula:**
```
mean = sum(x) / n
var = sum((x - mean)^2) / n
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

---

#### Task 1.2.2: Implement LayerNorm Backward Pass
**Priority:** Critical | **Complexity:** High

**Description:** Implement gradient computation for layer normalization.

**Acceptance Criteria:**
- [ ] Compute gradients w.r.t. input
- [ ] Compute gradients w.r.t. gamma, beta
- [ ] Handle batch dimension correctly
- [ ] Numerical gradient verification

**Files:**
- `NeuralNetwork/Layers/LayerNorm.cs`

---

#### Task 1.2.3: Implement LayerNorm CUDA Kernels
**Priority:** High | **Complexity:** High

**Description:** Create CUDA implementation of LayerNorm.

**Acceptance Criteria:**
- [ ] Parallel mean/variance computation
- [ ] Use shared memory for reductions
- [ ] Forward and backward kernels
- [ ] Match CPU output within epsilon

**Files:**
- `NeuralNetwork/Layers/Cuda/LayerNormCuda.cs`
- `NeuralNetwork/CU/LayerNormKernel.cu`

---

### 1.3 Dropout Layer

#### Task 1.3.1: Implement Dropout (CPU)
**Priority:** High | **Complexity:** Low

**Description:** Implement dropout with training/inference modes.

**Acceptance Criteria:**
- [ ] Random mask generation
- [ ] Inverted dropout (scale by 1/(1-p) during training)
- [ ] Pass-through during inference
- [ ] Store mask for backward pass

**Files:**
- `NeuralNetwork/Layers/Dropout.cs`

---

#### Task 1.3.2: Implement Dropout (CUDA)
**Priority:** Medium | **Complexity:** Medium

**Description:** CUDA dropout using cuRAND.

**Acceptance Criteria:**
- [ ] Use cuRAND for mask generation
- [ ] Efficient parallel implementation
- [ ] Same API as CPU version

**Files:**
- `NeuralNetwork/Layers/Cuda/DropoutCuda.cs`
- `NeuralNetwork/CU/DropoutKernel.cu`

---

## Phase 2: Attention Mechanism

### 2.1 Scaled Dot-Product Attention

#### Task 2.1.1: Implement Attention Score Computation
**Priority:** Critical | **Complexity:** Medium

**Description:** Compute attention scores as QK^T / sqrt(d_k).

**Acceptance Criteria:**
- [ ] Batched matrix multiply Q × K^T
- [ ] Scale by 1/sqrt(d_k)
- [ ] Output shape [batch, seq_len, seq_len]

**Files:**
- `NeuralNetwork/Layers/Attention/ScaledDotProductAttention.cs`

---

#### Task 2.1.2: Implement Causal Masking
**Priority:** Critical | **Complexity:** Low

**Description:** Apply causal mask to prevent attending to future tokens.

**Acceptance Criteria:**
- [ ] Generate lower-triangular mask
- [ ] Apply mask before softmax (-inf for masked positions)
- [ ] Support variable sequence lengths

**Files:**
- `NeuralNetwork/Layers/Attention/AttentionMask.cs`

**Mask Pattern:**
```
[1, -inf, -inf, -inf]
[1,    1, -inf, -inf]
[1,    1,    1, -inf]
[1,    1,    1,    1]
```

---

#### Task 2.1.3: Implement Attention Output Computation
**Priority:** Critical | **Complexity:** Medium

**Description:** Compute softmax(scores) × V.

**Acceptance Criteria:**
- [ ] Apply softmax along last dimension
- [ ] Multiply attention weights by values
- [ ] Output shape [batch, seq_len, d_v]

**Files:**
- `NeuralNetwork/Layers/Attention/ScaledDotProductAttention.cs`

---

#### Task 2.1.4: Implement Attention Backward Pass
**Priority:** Critical | **Complexity:** High

**Description:** Compute gradients through attention mechanism.

**Acceptance Criteria:**
- [ ] Gradient w.r.t. Q, K, V
- [ ] Handle softmax gradient correctly
- [ ] Account for masking in gradients

**Files:**
- `NeuralNetwork/Layers/Attention/ScaledDotProductAttention.cs`

---

#### Task 2.1.5: Implement Attention CUDA Kernel
**Priority:** High | **Complexity:** Very High

**Description:** Create fused CUDA kernel for attention.

**Acceptance Criteria:**
- [ ] Fused QK^T computation and scaling
- [ ] Masked softmax in single kernel
- [ ] Memory-efficient implementation
- [ ] Match CPU output within epsilon

**Files:**
- `NeuralNetwork/Layers/Cuda/AttentionCuda.cs`
- `NeuralNetwork/CU/AttentionKernel.cu`

---

### 2.2 Multi-Head Attention

#### Task 2.2.1: Implement Q, K, V Projections
**Priority:** Critical | **Complexity:** Medium

**Description:** Linear projections to create Q, K, V from input.

**Acceptance Criteria:**
- [ ] Three weight matrices W_Q, W_K, W_V
- [ ] Proper initialization (Xavier/Glorot)
- [ ] Support batched inputs

**Files:**
- `NeuralNetwork/Layers/Attention/MultiHeadAttention.cs`

---

#### Task 2.2.2: Implement Head Splitting
**Priority:** Critical | **Complexity:** Medium

**Description:** Split projected Q, K, V into multiple heads.

**Acceptance Criteria:**
- [ ] Reshape [B, S, D] → [B, H, S, D/H]
- [ ] Handle non-divisible cases with error
- [ ] Efficient memory layout for parallel computation

**Files:**
- `NeuralNetwork/Layers/Attention/MultiHeadAttention.cs`

---

#### Task 2.2.3: Implement Head Concatenation
**Priority:** Critical | **Complexity:** Low

**Description:** Concatenate attention outputs from all heads.

**Acceptance Criteria:**
- [ ] Reshape [B, H, S, D/H] → [B, S, D]
- [ ] Preserve proper order

**Files:**
- `NeuralNetwork/Layers/Attention/MultiHeadAttention.cs`

---

#### Task 2.2.4: Implement Output Projection
**Priority:** Critical | **Complexity:** Low

**Description:** Final linear projection after head concatenation.

**Acceptance Criteria:**
- [ ] W_O projection: d_model → d_model
- [ ] Proper initialization

**Files:**
- `NeuralNetwork/Layers/Attention/MultiHeadAttention.cs`

---

#### Task 2.2.5: Implement Multi-Head Attention Backward Pass
**Priority:** Critical | **Complexity:** Very High

**Description:** Full backward pass through multi-head attention.

**Acceptance Criteria:**
- [ ] Gradient through output projection
- [ ] Gradient through head concatenation
- [ ] Gradient through per-head attention
- [ ] Gradient through Q, K, V projections
- [ ] Accumulate weight gradients

**Files:**
- `NeuralNetwork/Layers/Attention/MultiHeadAttention.cs`

---

#### Task 2.2.6: Implement Multi-Head Attention CUDA
**Priority:** High | **Complexity:** Very High

**Description:** CUDA implementation of multi-head attention.

**Acceptance Criteria:**
- [ ] Parallel computation across heads
- [ ] Optimized memory access patterns
- [ ] Forward and backward passes

**Files:**
- `NeuralNetwork/Layers/Cuda/MultiHeadAttentionCuda.cs`
- `NeuralNetwork/CU/MultiHeadAttentionKernel.cu`

---

## Phase 3: Transformer Building Blocks

### 3.1 Positional Encoding

#### Task 3.1.1: Implement Sinusoidal Positional Encoding
**Priority:** Critical | **Complexity:** Low

**Description:** Fixed sinusoidal position embeddings.

**Acceptance Criteria:**
- [ ] Precompute encodings for max sequence length
- [ ] Correct sine/cosine pattern
- [ ] Add to token embeddings

**Files:**
- `NeuralNetwork/Layers/PositionalEncoding.cs`

**Formula:**
```
PE[pos, 2i] = sin(pos / 10000^(2i/d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
```

---

#### Task 3.1.2: Implement Learnable Positional Embeddings (Optional)
**Priority:** Low | **Complexity:** Low

**Description:** Alternative learnable position embeddings.

**Acceptance Criteria:**
- [ ] Embedding matrix [max_seq_len, d_model]
- [ ] Backward pass for learning

**Files:**
- `NeuralNetwork/Layers/LearnablePositionalEmbedding.cs`

---

### 3.2 Feed-Forward Network

#### Task 3.2.1: Implement GELU Activation
**Priority:** High | **Complexity:** Low

**Description:** Gaussian Error Linear Unit activation.

**Acceptance Criteria:**
- [ ] Forward: GELU(x) = x * Φ(x)
- [ ] Backward: proper gradient
- [ ] CUDA version

**Files:**
- `NeuralNetwork/Layers/Activations/GELU.cs`

**Approximation:**
```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
```

---

#### Task 3.2.2: Implement FFN Layer
**Priority:** Critical | **Complexity:** Medium

**Description:** Two-layer feed-forward network.

**Acceptance Criteria:**
- [ ] Linear1: d_model → d_ff
- [ ] GELU activation
- [ ] Linear2: d_ff → d_model
- [ ] Proper initialization

**Files:**
- `NeuralNetwork/Layers/FeedForward.cs`

---

#### Task 3.2.3: Implement FFN CUDA
**Priority:** Medium | **Complexity:** Medium

**Description:** CUDA-optimized FFN layer.

**Acceptance Criteria:**
- [ ] Use existing DenseCuda as base
- [ ] Add GELU CUDA kernel
- [ ] Fused operations where possible

**Files:**
- `NeuralNetwork/Layers/Cuda/FeedForwardCuda.cs`

---

### 3.3 Transformer Block

#### Task 3.3.1: Implement Residual Connections
**Priority:** Critical | **Complexity:** Low

**Description:** Add skip connections around attention and FFN.

**Acceptance Criteria:**
- [ ] output = input + sublayer(input)
- [ ] Handle gradient flow correctly

**Files:**
- `NeuralNetwork/Layers/TransformerBlock.cs`

---

#### Task 3.3.2: Implement Pre-LN Transformer Block
**Priority:** Critical | **Complexity:** Medium

**Description:** Complete Transformer block with Pre-LN architecture.

**Acceptance Criteria:**
- [ ] LayerNorm before attention
- [ ] Multi-head attention with residual
- [ ] LayerNorm before FFN
- [ ] FFN with residual
- [ ] Optional dropout

**Files:**
- `NeuralNetwork/Layers/TransformerBlock.cs`

**Architecture:**
```
x_norm = LayerNorm(x)
x = x + Dropout(MultiHeadAttention(x_norm))
x_norm = LayerNorm(x)
x = x + Dropout(FFN(x_norm))
```

---

#### Task 3.3.3: Implement Transformer Block Backward Pass
**Priority:** Critical | **Complexity:** High

**Description:** Full backward pass through Transformer block.

**Acceptance Criteria:**
- [ ] Gradient through all sublayers
- [ ] Proper residual gradient addition
- [ ] All parameter gradients computed

**Files:**
- `NeuralNetwork/Layers/TransformerBlock.cs`

---

## Phase 4: Complete Transformer Model

### 4.1 Transformer Decoder Stack

#### Task 4.1.1: Implement Stacked Transformer Blocks
**Priority:** Critical | **Complexity:** Medium

**Description:** Stack N Transformer blocks.

**Acceptance Criteria:**
- [ ] Configurable number of layers
- [ ] Sequential forward pass
- [ ] Sequential backward pass (reverse order)

**Files:**
- `NeuralNetwork/Models/TransformerDecoder.cs`

---

### 4.2 Transformer Language Model

#### Task 4.2.1: Create TransformerConfig
**Priority:** Critical | **Complexity:** Low

**Description:** Configuration class for model hyperparameters.

**Acceptance Criteria:**
- [ ] All architectural parameters
- [ ] Validation of parameter combinations
- [ ] Serialization for saving/loading

**Files:**
- `NeuralNetwork/Models/TransformerConfig.cs`

---

#### Task 4.2.2: Implement TransformerLM
**Priority:** Critical | **Complexity:** High

**Description:** Complete language model combining all components.

**Acceptance Criteria:**
- [ ] Token embedding layer
- [ ] Positional encoding
- [ ] N Transformer blocks
- [ ] Final layer norm
- [ ] Output projection to vocabulary
- [ ] Optional weight tying

**Files:**
- `NeuralNetwork/Models/TransformerLM.cs`

---

#### Task 4.2.3: Implement Weight Tying
**Priority:** Medium | **Complexity:** Low

**Description:** Share weights between token embedding and output projection.

**Acceptance Criteria:**
- [ ] Single weight matrix for both
- [ ] Proper gradient accumulation

**Files:**
- `NeuralNetwork/Models/TransformerLM.cs`

---

## Phase 5: Enhanced Text Processing

### 5.1 BPE Tokenizer

#### Task 5.1.1: Implement BPE Vocabulary Learning
**Priority:** High | **Complexity:** High

**Description:** Learn BPE merges from training corpus.

**Acceptance Criteria:**
- [ ] Count character/byte frequencies
- [ ] Iteratively merge most frequent pairs
- [ ] Build merge rules dictionary
- [ ] Configurable vocabulary size

**Files:**
- `NeuralNetwork/Processing/Text/BPETokenizer.cs`

---

#### Task 5.1.2: Implement BPE Encoding
**Priority:** High | **Complexity:** Medium

**Description:** Encode text to token IDs using learned BPE.

**Acceptance Criteria:**
- [ ] Apply merges in priority order
- [ ] Handle unknown characters
- [ ] Support batch encoding

**Files:**
- `NeuralNetwork/Processing/Text/BPETokenizer.cs`

---

#### Task 5.1.3: Implement BPE Decoding
**Priority:** High | **Complexity:** Low

**Description:** Decode token IDs back to text.

**Acceptance Criteria:**
- [ ] Reconstruct text from tokens
- [ ] Handle special tokens properly

**Files:**
- `NeuralNetwork/Processing/Text/BPETokenizer.cs`

---

#### Task 5.1.4: Implement Special Tokens
**Priority:** High | **Complexity:** Low

**Description:** Add support for PAD, BOS, EOS, UNK tokens.

**Acceptance Criteria:**
- [ ] Reserved token IDs
- [ ] Proper handling during encode/decode
- [ ] Configurable special tokens

**Files:**
- `NeuralNetwork/Processing/Text/SpecialTokens.cs`
- `NeuralNetwork/Processing/Text/Vocabulary.cs`

---

### 5.2 Data Pipeline

#### Task 5.2.1: Implement TextDataset
**Priority:** Medium | **Complexity:** Medium

**Description:** Dataset class for text data loading.

**Acceptance Criteria:**
- [ ] Load from file or memory
- [ ] Tokenize on-the-fly or precomputed
- [ ] Support shuffling
- [ ] Iterator interface

**Files:**
- `NeuralNetwork/Data/TextDataset.cs`

---

#### Task 5.2.2: Implement DataLoader
**Priority:** Medium | **Complexity:** Medium

**Description:** Batched data loading with shuffling.

**Acceptance Criteria:**
- [ ] Configurable batch size
- [ ] Optional shuffling per epoch
- [ ] Handle incomplete final batch
- [ ] Efficient memory usage

**Files:**
- `NeuralNetwork/Data/DataLoader.cs`

---

## Phase 6: Training Infrastructure

### 6.1 Learning Rate Scheduling

#### Task 6.1.1: Implement LR Scheduler Base
**Priority:** High | **Complexity:** Low

**Description:** Base class for learning rate schedulers.

**Acceptance Criteria:**
- [ ] Step method to update LR
- [ ] Get current LR method
- [ ] Integration with optimizers

**Files:**
- `NeuralNetwork/Optimizers/LRScheduler.cs`

---

#### Task 6.1.2: Implement Warmup + Cosine Decay
**Priority:** High | **Complexity:** Medium

**Description:** Standard Transformer learning rate schedule.

**Acceptance Criteria:**
- [ ] Linear warmup phase
- [ ] Cosine decay after warmup
- [ ] Configurable warmup steps and max steps

**Files:**
- `NeuralNetwork/Optimizers/WarmupCosineScheduler.cs`

---

### 6.2 Training Improvements

#### Task 6.2.1: Implement Gradient Accumulation
**Priority:** Medium | **Complexity:** Medium

**Description:** Accumulate gradients over multiple mini-batches.

**Acceptance Criteria:**
- [ ] Configurable accumulation steps
- [ ] Average gradients before update
- [ ] Works with existing optimizer

**Files:**
- `NeuralNetwork/Sequential.cs` or new `TransformerTrainer.cs`

---

#### Task 6.2.2: Fix Hardcoded Learning Rates
**Priority:** High | **Complexity:** Low

**Description:** Remove hardcoded 0.01f learning rates from layers.

**Acceptance Criteria:**
- [ ] Remove LR from Dense.Backward()
- [ ] Remove LR from CUDA kernels
- [ ] All updates go through Optimizer

**Files:**
- `NeuralNetwork/Layers/Dense.cs`
- `NeuralNetwork/Layers/Cuda/DenseCuda.cs`
- Various CUDA kernels

---

## Phase 7: Inference Optimization

### 7.1 KV-Cache

#### Task 7.1.1: Implement KV-Cache
**Priority:** High | **Complexity:** High

**Description:** Cache key-value pairs for autoregressive generation.

**Acceptance Criteria:**
- [ ] Store K, V tensors per layer
- [ ] Append new K, V during generation
- [ ] Memory-efficient implementation
- [ ] Works with multi-head attention

**Files:**
- `NeuralNetwork/Inference/KVCache.cs`

---

### 7.2 Advanced Sampling

#### Task 7.2.1: Implement Temperature Scaling
**Priority:** High | **Complexity:** Low

**Description:** Scale logits before softmax.

**Acceptance Criteria:**
- [ ] Divide logits by temperature
- [ ] Temperature = 1 is default
- [ ] Temperature < 1 makes distribution sharper

**Files:**
- `NeuralNetwork/Inference/Sampler.cs`

---

#### Task 7.2.2: Implement Top-K Sampling
**Priority:** High | **Complexity:** Medium

**Description:** Sample from top K most likely tokens.

**Acceptance Criteria:**
- [ ] Find top K logits
- [ ] Renormalize probabilities
- [ ] Sample from reduced distribution

**Files:**
- `NeuralNetwork/Inference/TopKSampler.cs`

---

#### Task 7.2.3: Implement Top-P (Nucleus) Sampling
**Priority:** High | **Complexity:** Medium

**Description:** Sample from smallest set with cumulative probability >= p.

**Acceptance Criteria:**
- [ ] Sort tokens by probability
- [ ] Find cutoff for cumulative probability
- [ ] Renormalize and sample

**Files:**
- `NeuralNetwork/Inference/TopPSampler.cs`

---

### 7.3 Transformer Generator

#### Task 7.3.1: Implement TransformerGenerator
**Priority:** High | **Complexity:** High

**Description:** Text generation with Transformer model.

**Acceptance Criteria:**
- [ ] Use KV-cache for efficiency
- [ ] Support all sampling methods
- [ ] Handle stop tokens
- [ ] Support streaming output

**Files:**
- `NeuralNetwork/Inference/TransformerGenerator.cs`

---

## Summary: Task Count by Phase

| Phase | Critical | High | Medium | Low | Total |
|-------|----------|------|--------|-----|-------|
| 1. Foundation | 6 | 3 | 2 | 0 | 11 |
| 2. Attention | 8 | 3 | 0 | 0 | 11 |
| 3. Building Blocks | 4 | 2 | 1 | 1 | 8 |
| 4. Complete Model | 3 | 0 | 1 | 0 | 4 |
| 5. Text Processing | 0 | 4 | 2 | 0 | 6 |
| 6. Training | 0 | 3 | 1 | 0 | 4 |
| 7. Inference | 0 | 5 | 1 | 0 | 6 |
| **Total** | **21** | **20** | **8** | **1** | **50** |

---

## Recommended Implementation Order

### Sprint 1: Core Tensor Infrastructure
1. Task 1.1.1: Tensor Class
2. Task 1.1.2: Batched MatMul (CPU)
3. Task 1.1.4: Transpose Operations
4. Task 1.1.5: Element-wise Operations

### Sprint 2: Layer Normalization & Dropout
5. Task 1.2.1: LayerNorm Forward (CPU)
6. Task 1.2.2: LayerNorm Backward
7. Task 1.3.1: Dropout (CPU)
8. Task 6.2.2: Fix Hardcoded Learning Rates

### Sprint 3: Attention Mechanism
9. Task 2.1.1: Attention Score Computation
10. Task 2.1.2: Causal Masking
11. Task 2.1.3: Attention Output Computation
12. Task 2.1.4: Attention Backward Pass

### Sprint 4: Multi-Head Attention
13. Task 2.2.1: Q, K, V Projections
14. Task 2.2.2: Head Splitting
15. Task 2.2.3: Head Concatenation
16. Task 2.2.4: Output Projection
17. Task 2.2.5: MHA Backward Pass

### Sprint 5: Building Blocks
18. Task 3.1.1: Sinusoidal Positional Encoding
19. Task 3.2.1: GELU Activation
20. Task 3.2.2: FFN Layer
21. Task 3.3.1: Residual Connections
22. Task 3.3.2: Pre-LN Transformer Block
23. Task 3.3.3: Transformer Block Backward

### Sprint 6: Complete Model
24. Task 4.2.1: TransformerConfig
25. Task 4.1.1: Stacked Transformer Blocks
26. Task 4.2.2: TransformerLM
27. Task 4.2.3: Weight Tying

### Sprint 7: CUDA Acceleration
28. Task 1.1.3: Batched MatMul (CUDA)
29. Task 1.2.3: LayerNorm CUDA Kernels
30. Task 1.3.2: Dropout (CUDA)
31. Task 2.1.5: Attention CUDA Kernel
32. Task 2.2.6: Multi-Head Attention CUDA

### Sprint 8: Text Processing
33. Task 5.1.4: Special Tokens
34. Task 5.1.1: BPE Vocabulary Learning
35. Task 5.1.2: BPE Encoding
36. Task 5.1.3: BPE Decoding
37. Task 5.2.1: TextDataset
38. Task 5.2.2: DataLoader

### Sprint 9: Training Infrastructure
39. Task 6.1.1: LR Scheduler Base
40. Task 6.1.2: Warmup + Cosine Decay
41. Task 6.2.1: Gradient Accumulation

### Sprint 10: Inference
42. Task 7.2.1: Temperature Scaling
43. Task 7.2.2: Top-K Sampling
44. Task 7.2.3: Top-P Sampling
45. Task 7.1.1: KV-Cache
46. Task 7.3.1: TransformerGenerator

---

*This task breakdown is designed to be completed incrementally. Each sprint delivers a testable piece of functionality.*
