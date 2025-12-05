# Transformer Architecture Migration Plan

## Executive Summary

This document outlines the migration path from the current LSTM/RNN-based text training architecture to a modern Transformer-based approach. The goal is to implement the core Transformer components ("Attention Is All You Need" - Vaswani et al., 2017) entirely in .NET with CUDA acceleration, maintaining minimal external dependencies.

---

## Current Architecture Analysis

### What We Have
- **Sequential Model API** - Keras-like interface (keep)
- **Layer Abstraction** - `Layer` base class with `Call()`, `Backward()`, `Build()` (extend)
- **CUDA Infrastructure** - ManagedCuda integration, PTX kernel loading (reuse)
- **Optimizers** - Adam, SGD implementations (keep)
- **Loss Functions** - CategoricalCrossentropy (keep)
- **Text Preprocessing** - Basic tokenization (upgrade)

### What Needs to Change
| Component | Current | Target |
|-----------|---------|--------|
| Sequence Processing | LSTM/GRU (recurrent) | Self-Attention (parallel) |
| Position Information | Implicit in recurrence | Positional Encoding |
| Token Processing | Simple word tokenization | BPE/WordPiece tokenization |
| Architecture | Single layer types | Multi-Head Attention + FFN blocks |
| Normalization | None | LayerNorm |
| Regularization | None | Dropout |

---

## Transformer Architecture Overview

```
Input Tokens
     ↓
┌─────────────────────────────────────────┐
│  Token Embedding + Positional Encoding  │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│         Transformer Block (×N)          │
│  ┌───────────────────────────────────┐  │
│  │   Multi-Head Self-Attention       │  │
│  │   + Residual Connection           │  │
│  │   + Layer Normalization           │  │
│  └───────────────────────────────────┘  │
│                   ↓                     │
│  ┌───────────────────────────────────┐  │
│  │   Feed-Forward Network            │  │
│  │   + Residual Connection           │  │
│  │   + Layer Normalization           │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│      Output Projection (Vocabulary)     │
└─────────────────────────────────────────┘
     ↓
Output Logits
```

---

## Migration Milestones

### Phase 1: Foundation Components
**Goal:** Build the mathematical primitives required for attention

#### Milestone 1.1: Tensor Operations Library
- [ ] Create `Tensor` class with proper shape management
- [ ] Implement batched matrix multiplication (BMM)
- [ ] Implement transpose operations (batch-aware)
- [ ] Implement element-wise operations (add, multiply, scale)
- [ ] Add broadcasting support
- [ ] Create CUDA kernels for all tensor operations

**Files to Create:**
```
NeuralNetwork/
├── Tensors/
│   ├── Tensor.cs              # Core tensor class
│   ├── TensorOperations.cs    # CPU implementations
│   └── TensorOperationsCuda.cs # GPU implementations
├── CU/
│   ├── TensorKernels.cu       # CUDA tensor operations
│   └── TensorKernels.ptx      # Compiled kernels
```

**Key Operations:**
```csharp
// Required tensor operations
Tensor MatMul(Tensor a, Tensor b);           // [B,M,K] × [B,K,N] → [B,M,N]
Tensor Transpose(Tensor t, int dim1, int dim2);
Tensor Scale(Tensor t, float scalar);
Tensor Add(Tensor a, Tensor b);              // With broadcasting
Tensor Concat(Tensor[] tensors, int axis);
Tensor Split(Tensor t, int numSplits, int axis);
```

#### Milestone 1.2: Layer Normalization
- [ ] Implement LayerNorm CPU version
- [ ] Implement LayerNorm CUDA kernel
- [ ] Add learnable parameters (gamma, beta)
- [ ] Implement backward pass

**Files to Create:**
```
NeuralNetwork/
├── Layers/
│   ├── LayerNorm.cs
│   └── Cuda/
│       └── LayerNormCuda.cs
├── CU/
│   ├── LayerNormKernel.cu
│   └── LayerNormKernel.ptx
```

**Formula:**
```
y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
```

#### Milestone 1.3: Dropout Layer
- [ ] Implement Dropout with training/inference modes
- [ ] Implement inverted dropout (scale during training)
- [ ] Add CUDA version with cuRAND

**Files to Create:**
```
NeuralNetwork/
├── Layers/
│   ├── Dropout.cs
│   └── Cuda/
│       └── DropoutCuda.cs
```

---

### Phase 2: Attention Mechanism
**Goal:** Implement the core attention computation

#### Milestone 2.1: Scaled Dot-Product Attention
- [ ] Implement Q, K, V projections
- [ ] Implement attention score computation: `softmax(QK^T / sqrt(d_k))`
- [ ] Implement causal masking (for decoder/autoregressive)
- [ ] Implement attention output computation
- [ ] Create CUDA kernels for fused attention

**Files to Create:**
```
NeuralNetwork/
├── Layers/
│   ├── Attention/
│   │   ├── ScaledDotProductAttention.cs
│   │   └── AttentionMask.cs
│   └── Cuda/
│       └── AttentionCuda.cs
├── CU/
│   ├── AttentionKernel.cu
│   └── AttentionKernel.ptx
```

**Key Algorithm:**
```csharp
// Scaled Dot-Product Attention
// Q: [batch, seq_len, d_k]
// K: [batch, seq_len, d_k]
// V: [batch, seq_len, d_v]

scores = MatMul(Q, Transpose(K)) / sqrt(d_k);  // [batch, seq_len, seq_len]
if (causal_mask)
    scores = ApplyMask(scores, mask);           // -inf for future tokens
attention_weights = Softmax(scores, axis=-1);
output = MatMul(attention_weights, V);          // [batch, seq_len, d_v]
```

#### Milestone 2.2: Multi-Head Attention
- [ ] Implement head splitting and concatenation
- [ ] Implement parallel attention computation per head
- [ ] Implement output projection
- [ ] Optimize CUDA kernels for multi-head parallelism

**Files to Create:**
```
NeuralNetwork/
├── Layers/
│   ├── Attention/
│   │   └── MultiHeadAttention.cs
│   └── Cuda/
│       └── MultiHeadAttentionCuda.cs
├── CU/
│   ├── MultiHeadAttentionKernel.cu
│   └── MultiHeadAttentionKernel.ptx
```

**Architecture:**
```
Input [batch, seq_len, d_model]
    ↓
┌───────────────────────────────────────────────┐
│  Linear Projections (W_Q, W_K, W_V)           │
│  Q, K, V each: [batch, seq_len, d_model]      │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│  Split into num_heads                         │
│  [batch, num_heads, seq_len, d_k]             │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│  Scaled Dot-Product Attention (per head)      │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│  Concatenate heads                            │
│  [batch, seq_len, d_model]                    │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│  Output Projection (W_O)                      │
└───────────────────────────────────────────────┘
```

---

### Phase 3: Transformer Building Blocks
**Goal:** Assemble attention into complete Transformer layers

#### Milestone 3.1: Positional Encoding
- [ ] Implement sinusoidal positional encoding (fixed)
- [ ] Implement learnable positional embeddings (optional)
- [ ] Support variable sequence lengths

**Files to Create:**
```
NeuralNetwork/
├── Layers/
│   ├── PositionalEncoding.cs
│   └── LearnablePositionalEmbedding.cs
```

**Sinusoidal Formula:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### Milestone 3.2: Feed-Forward Network (FFN)
- [ ] Implement two-layer FFN with expansion
- [ ] Support GELU activation (preferred) or ReLU
- [ ] Add CUDA optimized version

**Files to Create:**
```
NeuralNetwork/
├── Layers/
│   ├── FeedForward.cs
│   └── Cuda/
│       └── FeedForwardCuda.cs
├── Layers/Activations/
│   └── GELU.cs
```

**Architecture:**
```
FFN(x) = Linear2(GELU(Linear1(x)))

Linear1: d_model → d_ff (typically 4 * d_model)
Linear2: d_ff → d_model
```

#### Milestone 3.3: Transformer Block
- [ ] Combine Multi-Head Attention + FFN
- [ ] Implement residual connections
- [ ] Implement Pre-LN vs Post-LN variants
- [ ] Add dropout at residual connections

**Files to Create:**
```
NeuralNetwork/
├── Layers/
│   └── TransformerBlock.cs
```

**Block Structure (Pre-LN variant - more stable):**
```
x_norm = LayerNorm(x)
attn_out = MultiHeadAttention(x_norm, x_norm, x_norm)
x = x + Dropout(attn_out)

x_norm = LayerNorm(x)
ffn_out = FeedForward(x_norm)
x = x + Dropout(ffn_out)
```

---

### Phase 4: Complete Transformer Model
**Goal:** Create the full decoder-only Transformer for text generation

#### Milestone 4.1: Transformer Decoder Stack
- [ ] Stack N Transformer blocks
- [ ] Implement weight sharing options
- [ ] Add gradient checkpointing (memory optimization)

**Files to Create:**
```
NeuralNetwork/
├── Models/
│   └── TransformerDecoder.cs
```

#### Milestone 4.2: Transformer Language Model
- [ ] Combine embedding + positional encoding + decoder + output projection
- [ ] Implement weight tying (embedding ↔ output projection)
- [ ] Create model configuration class

**Files to Create:**
```
NeuralNetwork/
├── Models/
│   ├── TransformerLM.cs
│   └── TransformerConfig.cs
```

**Complete Architecture:**
```csharp
public class TransformerLM
{
    TransformerConfig Config;
    EmbeddingCuda TokenEmbedding;      // vocab_size → d_model
    PositionalEncoding PosEncoding;     // seq_len → d_model
    TransformerBlock[] Blocks;          // N blocks
    LayerNorm FinalNorm;
    Dense OutputProjection;             // d_model → vocab_size

    public float[,] Forward(int[] tokens)
    {
        var x = TokenEmbedding.Call(tokens);
        x = Add(x, PosEncoding.GetEncoding(tokens.Length));

        foreach (var block in Blocks)
            x = block.Forward(x);

        x = FinalNorm.Forward(x);
        return OutputProjection.Call(x);  // logits
    }
}
```

---

### Phase 5: Enhanced Text Processing
**Goal:** Upgrade tokenization for Transformer models

#### Milestone 5.1: BPE Tokenizer
- [ ] Implement Byte-Pair Encoding algorithm
- [ ] Support vocabulary learning from corpus
- [ ] Implement encode/decode methods
- [ ] Add special tokens (PAD, BOS, EOS, UNK)

**Files to Create:**
```
NeuralNetwork/
├── Processing/Text/
│   ├── BPETokenizer.cs
│   ├── Vocabulary.cs
│   └── SpecialTokens.cs
```

#### Milestone 5.2: Data Pipeline
- [ ] Implement efficient data loading
- [ ] Implement sequence packing (optional)
- [ ] Add data augmentation options

**Files to Create:**
```
NeuralNetwork/
├── Data/
│   ├── TextDataset.cs
│   └── DataLoader.cs
```

---

### Phase 6: Training Infrastructure
**Goal:** Optimize training for Transformers

#### Milestone 6.1: Learning Rate Scheduling
- [ ] Implement warmup + cosine decay
- [ ] Implement linear decay
- [ ] Add scheduler base class

**Files to Create:**
```
NeuralNetwork/
├── Optimizers/
│   ├── LRScheduler.cs
│   ├── WarmupCosineScheduler.cs
│   └── LinearWarmupScheduler.cs
```

**Warmup Schedule:**
```
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    lr = base_lr * cosine_decay(step - warmup_steps)
```

#### Milestone 6.2: Gradient Accumulation
- [ ] Implement gradient accumulation for large effective batch sizes
- [ ] Add gradient clipping improvements

**Changes to:**
```
NeuralNetwork/Sequential.cs (or new TransformerTrainer.cs)
```

#### Milestone 6.3: Mixed Precision Training (Optional)
- [ ] Implement FP16 forward pass
- [ ] Implement loss scaling
- [ ] Add FP32 master weights

---

### Phase 7: Inference Optimization
**Goal:** Fast and flexible text generation

#### Milestone 7.1: KV-Cache
- [ ] Implement key-value caching for autoregressive generation
- [ ] Optimize memory layout for cache

**Files to Create:**
```
NeuralNetwork/
├── Inference/
│   └── KVCache.cs
```

#### Milestone 7.2: Advanced Sampling
- [ ] Implement temperature scaling
- [ ] Implement top-k sampling
- [ ] Implement top-p (nucleus) sampling
- [ ] Implement beam search

**Files to Create:**
```
NeuralNetwork/
├── Inference/
│   ├── Sampler.cs
│   ├── TopKSampler.cs
│   ├── TopPSampler.cs
│   └── BeamSearch.cs
```

#### Milestone 7.3: Transformer Text Generator
- [ ] Create new generator utilizing KV-cache
- [ ] Support streaming generation
- [ ] Add stop token handling

**Files to Create:**
```
NeuralNetwork/
├── Inference/
│   └── TransformerGenerator.cs
```

---

## Implementation Order & Dependencies

```
Phase 1 (Foundation)
├── 1.1 Tensor Operations ──────────────────────┐
├── 1.2 LayerNorm ──────────────────────────────┤
└── 1.3 Dropout ────────────────────────────────┤
                                                ↓
Phase 2 (Attention)                             │
├── 2.1 Scaled Dot-Product Attention ←──────────┤
└── 2.2 Multi-Head Attention ←──────────────────┘
                    ↓
Phase 3 (Building Blocks)
├── 3.1 Positional Encoding
├── 3.2 Feed-Forward Network
└── 3.3 Transformer Block ←── (requires 2.2, 3.2, 1.2, 1.3)
                    ↓
Phase 4 (Complete Model)
├── 4.1 Transformer Decoder Stack
└── 4.2 Transformer Language Model
                    ↓
Phase 5 (Text Processing) ──── Can be done in parallel with Phase 3-4
├── 5.1 BPE Tokenizer
└── 5.2 Data Pipeline
                    ↓
Phase 6 (Training)
├── 6.1 LR Scheduling
├── 6.2 Gradient Accumulation
└── 6.3 Mixed Precision (optional)
                    ↓
Phase 7 (Inference)
├── 7.1 KV-Cache
├── 7.2 Advanced Sampling
└── 7.3 Transformer Generator
```

---

## CUDA Kernel Requirements

### New Kernels Needed

| Kernel | Purpose | Priority |
|--------|---------|----------|
| `BatchedMatMul` | [B,M,K]×[B,K,N] multiplication | Critical |
| `SoftmaxWithMask` | Masked softmax for attention | Critical |
| `LayerNorm` | Layer normalization forward/backward | Critical |
| `AttentionFused` | Fused QKV attention (memory efficient) | High |
| `GELU` | Gaussian Error Linear Unit activation | High |
| `Dropout` | Random dropout with cuRAND | High |
| `PositionalEncoding` | Compute sinusoidal encodings | Medium |
| `KVCacheUpdate` | Efficient KV-cache operations | Medium |

### Kernel Optimization Strategies
1. **Fused Operations** - Combine LayerNorm + Attention + Dropout
2. **Memory Coalescing** - Ensure aligned memory access patterns
3. **Shared Memory** - Use for attention score computation
4. **Tensor Cores** - FP16 matrix multiply (if supported)

---

## Configuration System

```csharp
public class TransformerConfig
{
    // Model architecture
    public int VocabSize { get; set; } = 32000;
    public int MaxSeqLen { get; set; } = 512;
    public int NumLayers { get; set; } = 6;
    public int NumHeads { get; set; } = 8;
    public int EmbeddingDim { get; set; } = 512;
    public int FFNDim { get; set; } = 2048;  // Usually 4x EmbeddingDim

    // Regularization
    public float DropoutRate { get; set; } = 0.1f;
    public float AttentionDropout { get; set; } = 0.1f;

    // Training
    public float LearningRate { get; set; } = 1e-4f;
    public int WarmupSteps { get; set; } = 4000;
    public float WeightDecay { get; set; } = 0.01f;

    // Options
    public bool TieEmbeddings { get; set; } = true;
    public bool UseCausalMask { get; set; } = true;
    public string NormType { get; set; } = "pre";  // "pre" or "post"
}
```

---

## Testing Strategy

### Unit Tests for Each Component
```
NeuralNetwork.Tests/
├── Tensors/
│   └── TensorOperationsTests.cs
├── Layers/
│   ├── LayerNormTests.cs
│   ├── AttentionTests.cs
│   ├── MultiHeadAttentionTests.cs
│   └── TransformerBlockTests.cs
├── Models/
│   └── TransformerLMTests.cs
└── Processing/
    └── BPETokenizerTests.cs
```

### Validation Approach
1. Compare CPU vs CUDA outputs (should match within epsilon)
2. Gradient checking with numerical differentiation
3. Compare outputs with reference implementations (PyTorch)
4. Benchmark performance (tokens/second)

---

## Migration Path for Existing Code

### Backward Compatibility
- Keep existing `Sequential` model working
- LSTM/GRU layers remain available for comparison
- Existing training examples continue to work

### New Training Example
```csharp
// New Transformer-based text generation
var config = new TransformerConfig
{
    VocabSize = tokenizer.VocabSize,
    MaxSeqLen = 256,
    NumLayers = 4,
    NumHeads = 4,
    EmbeddingDim = 256,
    FFNDim = 1024
};

var model = new TransformerLM(config);
var trainer = new TransformerTrainer(model, new Adam(config.LearningRate));
trainer.Train(dataset, epochs: 10);

var generator = new TransformerGenerator(model, tokenizer);
string output = generator.Generate("Once upon a time", maxTokens: 100);
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory consumption | High | Gradient checkpointing, smaller batches |
| CUDA complexity | High | Thorough testing, CPU fallback |
| Training instability | Medium | Pre-LN, proper initialization, warmup |
| Performance regression | Medium | Benchmark against LSTM baseline |
| Scope creep | Medium | Strict milestone adherence |

---

## Success Metrics

1. **Functional** - Model trains and generates coherent text
2. **Performance** - Faster training than LSTM for same quality
3. **Memory** - Fits in 8GB VRAM for reasonable model sizes
4. **Quality** - Lower perplexity than LSTM baseline
5. **Code Quality** - Clean, documented, testable code

---

## Appendix A: Reference Papers

1. "Attention Is All You Need" (Vaswani et al., 2017) - Original Transformer
2. "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) - GPT-2
3. "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020) - Pre-LN
4. "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016) - BPE

---

## Appendix B: File Structure After Migration

```
NeuralTrainer-NET/
├── NET_Keras/
│   └── Program.cs                    # Updated with Transformer examples
├── NeuralNetwork/
│   ├── Sequential.cs                 # Keep (backward compat)
│   ├── Layer.cs                      # Keep (base class)
│   │
│   ├── Tensors/                      # NEW
│   │   ├── Tensor.cs
│   │   ├── TensorOperations.cs
│   │   └── TensorOperationsCuda.cs
│   │
│   ├── Layers/
│   │   ├── Dense.cs                  # Keep
│   │   ├── Embedding.cs              # Keep
│   │   ├── LSTM.cs                   # Keep (backward compat)
│   │   ├── LayerNorm.cs              # NEW
│   │   ├── Dropout.cs                # NEW
│   │   ├── FeedForward.cs            # NEW
│   │   ├── PositionalEncoding.cs     # NEW
│   │   ├── TransformerBlock.cs       # NEW
│   │   │
│   │   ├── Attention/                # NEW
│   │   │   ├── ScaledDotProductAttention.cs
│   │   │   ├── MultiHeadAttention.cs
│   │   │   └── AttentionMask.cs
│   │   │
│   │   ├── Activations/
│   │   │   ├── Activations.cs        # Keep
│   │   │   └── GELU.cs               # NEW
│   │   │
│   │   └── Cuda/
│   │       ├── DenseCuda.cs          # Keep
│   │       ├── EmbeddingCuda.cs      # Keep
│   │       ├── LSTMCuda.cs           # Keep
│   │       ├── LayerNormCuda.cs      # NEW
│   │       ├── DropoutCuda.cs        # NEW
│   │       ├── AttentionCuda.cs      # NEW
│   │       └── MultiHeadAttentionCuda.cs  # NEW
│   │
│   ├── Models/                       # NEW
│   │   ├── TransformerDecoder.cs
│   │   ├── TransformerLM.cs
│   │   └── TransformerConfig.cs
│   │
│   ├── Optimizers/
│   │   ├── Adam.cs                   # Keep
│   │   ├── SGD.cs                    # Keep
│   │   ├── LRScheduler.cs            # NEW
│   │   └── WarmupCosineScheduler.cs  # NEW
│   │
│   ├── Losses/
│   │   ├── CategoricalCrossentropy.cs  # Keep
│   │   └── MeanSquaredError.cs         # Keep
│   │
│   ├── Processing/
│   │   └── Text/
│   │       ├── TextPreProcessing.cs    # Keep
│   │       ├── BPETokenizer.cs         # NEW
│   │       ├── Vocabulary.cs           # NEW
│   │       └── SpecialTokens.cs        # NEW
│   │
│   ├── Data/                         # NEW
│   │   ├── TextDataset.cs
│   │   └── DataLoader.cs
│   │
│   ├── Inference/
│   │   ├── TextGenerator.cs          # Keep
│   │   ├── TransformerGenerator.cs   # NEW
│   │   ├── KVCache.cs                # NEW
│   │   ├── Sampler.cs                # NEW
│   │   ├── TopKSampler.cs            # NEW
│   │   └── TopPSampler.cs            # NEW
│   │
│   └── CU/
│       ├── DenseKernel.cu            # Keep
│       ├── LSTMKernel.cu             # Keep
│       ├── TensorKernels.cu          # NEW
│       ├── LayerNormKernel.cu        # NEW
│       ├── AttentionKernel.cu        # NEW
│       ├── DropoutKernel.cu          # NEW
│       └── *.ptx                     # Compiled kernels
│
└── NeuralNetwork.Tests/              # NEW (recommended)
    └── ...
```

---

*Document Version: 1.0*
*Created: December 2024*
*Last Updated: December 2024*
