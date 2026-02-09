# NeuralTrainer-NET

A C#/.NET implementation of a machine learning framework with CUDA GPU acceleration. "If Python can do it, I can do it too!"

## Project Vision

Build a comprehensive suite of AI training tools supporting multiple modalities:
- Text generation (implemented - Transformer LM, Sequential LSTM)
- Image classification (implemented - Conv2D + MaxPool2D + Dense)
- Text-to-speech (implemented - Tacotron-style encoder-decoder with CPU and GPU)
- Audio processing (partially implemented - WAV loading, mel spectrograms, microphone input)
- Audio transcription / ASR (planned)
- Audio generation / vocoder (planned)
- Speech-to-speech (planned)

## Architecture Overview

```
NeuralTrainer-NET/
├── NET_Keras/                             # CLI Application (entry point)
│   ├── NET_Keras-CMD.csproj               # .NET 9.0 executable project
│   ├── Program.cs                         # Training examples and main entry
│   ├── training_text.txt                  # Sample training data
│   └── tokenizer/                         # BPE tokenizer data
│       ├── tokenizer_config.json
│       ├── vocab.json
│       └── merges.txt
│
├── NET_Keras_CUDA/                        # Alternate CUDA-focused CLI project
│   └── NET_Keras_CUDA/
│       ├── NET_Keras_CUDA.csproj
│       └── Program.cs
│
├── NeuralNetwork/                         # Core ML Framework (class library)
│   ├── NeuralNetwork.csproj               # .NET 9.0 library, ManagedCuda 10.0.0
│   ├── Layer.cs                           # Abstract base layer class
│   ├── Sequential.cs                      # Keras-like Sequential model API
│   ├── LearningRateScheduler.cs           # LR scheduling
│   │
│   ├── Layers/                            # Layer implementations (CPU)
│   │   ├── Dense.cs                       # Fully connected layer
│   │   ├── Embedding.cs                   # Token embedding
│   │   ├── LSTM.cs                        # LSTM recurrent cell
│   │   ├── Conv2D.cs                      # 2D convolution
│   │   ├── MaxPool2D.cs                   # Max pooling
│   │   ├── Dropout.cs                     # Dropout regularization
│   │   ├── LayerNorm.cs                   # Layer normalization
│   │   ├── FeedForward.cs                 # FFN/MLP block (2x Dense)
│   │   ├── TransformerBlock.cs            # Transformer decoder block (pre-LN)
│   │   ├── PositionalEncoding.cs          # Sinusoidal positional encoding
│   │   │
│   │   ├── Activations/                   # Activation functions
│   │   │   ├── Activations.cs             # ReLU, Sigmoid, Tanh, Linear, Softmax
│   │   │   ├── GELU.cs                    # GELU activation
│   │   │   └── ActivationsCuda.cs         # GPU-accelerated activations
│   │   │
│   │   ├── Attention/                     # Attention mechanisms
│   │   │   ├── MultiHeadAttention.cs      # Multi-head attention layer
│   │   │   ├── ScaledDotProductAttention.cs # Scaled dot-product attention
│   │   │   └── AttentionMask.cs           # Causal masking utilities
│   │   │
│   │   └── Cuda/                          # GPU-accelerated layer versions
│   │       ├── DenseCuda.cs               # GPU dense (DenseKernel.ptx)
│   │       ├── DenseCudaOptimized.cs      # Tiled shared memory variant
│   │       ├── LSTMCuda.cs                # GPU LSTM (LSTMKernel.ptx)
│   │       ├── GRUCuda.cs                 # GPU GRU (GRUKernel.ptx)
│   │       ├── EmbeddingCuda.cs           # GPU embedding (EmbeddingKernel.ptx)
│   │       ├── DropoutCuda.cs             # GPU dropout
│   │       ├── LayerNormCuda.cs           # GPU layer norm (LayerNormKernel.ptx)
│   │       ├── FeedForwardCuda.cs         # GPU FFN (FeedForwardKernel.ptx)
│   │       ├── MultiHeadAttentionCuda.cs  # GPU MHA (AttentionKernel.ptx)
│   │       ├── ScaledDotProductAttentionCuda.cs
│   │       └── TransformerBlockCuda.cs    # Full GPU transformer block
│   │
│   ├── Models/                            # High-level model architectures
│   │   ├── TransformerConfig.cs           # Transformer hyperparameters
│   │   ├── TransformerLM.cs               # GPT-style decoder-only LM (CPU)
│   │   ├── TransformerLMCuda.cs           # GPU version of TransformerLM
│   │   └── TTS/                           # Text-to-Speech
│   │       ├── TTSConfig.cs               # TTS hyperparameters
│   │       ├── TTSModel.cs               # Tacotron-style TTS (CPU)
│   │       └── TTSModelCuda.cs            # GPU TTS with backward pass
│   │
│   ├── Optimizers/                        # Training optimizers
│   │   ├── Optimizer.cs                   # Abstract base (Update, Update4D)
│   │   ├── Adam.cs                        # Adam (2D & 4D tensor support)
│   │   └── SGD.cs                         # Stochastic gradient descent
│   │
│   ├── Losses/                            # Loss functions
│   │   ├── Loss.cs                        # Abstract base (Calculate, Calculate4D)
│   │   ├── MeanSquaredError.cs            # MSE loss
│   │   └── CategoricalCrossentropy.cs     # Softmax cross-entropy
│   │
│   ├── Processing/                        # Data preprocessing
│   │   ├── Text/
│   │   │   ├── TextPreProcessing.cs       # Tokenization, padding, sequences
│   │   │   ├── TextDataset.cs             # Text dataset utilities
│   │   │   └── BPETokenizer.cs            # Byte-pair encoding tokenizer
│   │   └── Audio/
│   │       ├── AudioPreProcessing.cs      # WAV loading, resampling, normalization
│   │       ├── AudioDataset.cs            # Audio dataset utilities
│   │       ├── MelSpectrogram.cs          # Mel-frequency spectrogram (CPU)
│   │       ├── MelSpectrogramCuda.cs      # GPU mel spectrograms (AudioKernel.ptx)
│   │       └── MicrophoneStream.cs        # Real-time microphone input
│   │
│   ├── Inference/                         # Model inference utilities
│   │   ├── TextGenerator.cs               # Text generation with sampling
│   │   ├── Sampler.cs                     # Greedy, temperature, top-k, top-p
│   │   ├── KVCache.cs                     # Key-value cache for transformers
│   │   └── Audio/
│   │       └── SpeechSynthesizer.cs       # TTS synthesis utilities
│   │
│   ├── Cuda/                              # GPU infrastructure
│   │   ├── CudaAccelerator.cs             # Singleton GPU context manager
│   │   ├── CudaMemoryPool.cs             # Memory pooling (2-4x speedup)
│   │   ├── CudaKernelCache.cs             # Kernel caching (1.1-1.2x speedup)
│   │   ├── CudaStreamManager.cs           # Multi-stream parallelism (1.3-1.8x)
│   │   └── CudaBenchmark.cs              # GPU benchmarking utilities
│   │
│   ├── CU/                               # CUDA C++ kernels (.cu source + .ptx compiled)
│   │   ├── DenseKernel.cu/ptx             # MatMul, fused MatMul+Bias+ReLU
│   │   ├── LSTMKernel.cu/ptx             # LSTM forward/backward
│   │   ├── LSTMKernelV2.cu/ptx           # Optimized LSTM variant
│   │   ├── EmbeddingKernel.cu/ptx         # Embedding lookup
│   │   ├── LayerNormKernel.cu/ptx         # Layer normalization
│   │   ├── ActivationsKernel.cu/ptx       # ReLU, Sigmoid, Tanh, GELU
│   │   ├── FeedForwardKernel.cu/ptx       # FFN/MLP kernels
│   │   ├── AttentionKernel.cu/ptx         # Batched MatMul, softmax, scaling
│   │   ├── GRUKernel.cu/ptx              # GRU cell kernels
│   │   ├── AudioKernel.cu/ptx            # Audio processing / mel spectrograms
│   │   └── TTSKernel.cu/ptx              # TTS-specific kernels
│   │
│   ├── Training/                          # Training utilities
│   │   ├── Checkpoint.cs                  # Model checkpointing with metadata
│   │   ├── GradientAccumulator.cs         # Gradient accumulation for larger batches
│   │   └── LearningRateScheduler.cs       # LR scheduling strategies
│   │
│   ├── Tensors/                           # Tensor abstraction
│   │   ├── Tensor.cs                      # Multi-dimensional tensor (row-major)
│   │   └── TensorOperations.cs            # Tensor math operations
│   │
│   ├── Initializers/
│   │   └── Initializers.cs                # Glorot, Xavier, He, Normal initialization
│   │
│   ├── Ext/                               # Extension methods
│   │   ├── SaveSequential.cs              # Model serialization
│   │   └── LoadSequential.cs              # Model deserialization
│   │
│   └── SerializationHelper/
│       └── ArrayHelpers.cs                # Array serialization utilities
│
├── NeuralNetwork.Tests/                   # xUnit test project
│   ├── NeuralNetwork.Tests.csproj         # xUnit 2.6.2, .NET 9.0, x64
│   ├── Layers/
│   │   └── DenseTests.cs                  # Dense layer unit tests
│   ├── Cuda/
│   │   ├── CudaAvailabilityTests.cs       # GPU availability checks
│   │   └── TTS/
│   │       ├── TTSKernelLoadTests.cs      # TTS kernel loading tests
│   │       └── Conv1DKernelTests.cs       # Conv1D kernel tests
│   └── Helpers/
│       ├── CudaTestFixture.cs             # Base fixture for CUDA tests
│       ├── GradientChecker.cs             # Numerical gradient verification
│       ├── MatrixAssert.cs                # Matrix comparison assertions
│       └── TestDataBuilder.cs             # Test data generation
│
├── docs/                                  # Additional documentation
│   ├── TRANSFORMER_MIGRATION.md
│   └── TRANSFORMER_TASKS.md
│
├── NET_Keras.sln                          # Visual Studio solution
├── CLAUDE.md                              # This file
└── IMPLEMENTATION_SUMMARY.md              # Recent implementation notes
```

## Technology Stack

- **Language**: C# (.NET 9.0)
- **GPU**: CUDA via ManagedCuda v10.0.0 (CUDA Toolkit v13.0)
- **Build**: MSBuild / Visual Studio 2022+
- **Platform**: x64 (required for CUDA)
- **Testing**: xUnit 2.6.2, coverlet for coverage
- **Serialization**: XML + custom binary serialization

## Key Classes

### Core Framework
- `Layer` - Abstract base class for all layers. Methods: `Build()`, `Call()`, `Backward()`, `GetOutputShape()`
- `Sequential` - Keras-like model API. Methods: `Add()`, `Compile()`, `Fit()`, `Predict()`, `TrainOnBatch()`
- `Optimizer` - Abstract base for optimizers. Methods: `Update()`, `Update4D()`
- `Loss` - Abstract base for loss functions. Methods: `Calculate()`, `Calculate4D()`

### High-Level Models
- `TransformerLM` - GPT-style decoder-only language model with embedding, positional encoding, N transformer blocks, and output projection
- `TransformerLMCuda` - GPU version with memory pooling and stream parallelism
- `TTSModel` / `TTSModelCuda` - Tacotron-style TTS: text embedding → encoder convolutions → location-sensitive attention → LSTM decoder → mel projection → postnet

### CUDA Infrastructure
- `CudaAccelerator` - Singleton GPU context manager (access via `CudaAccelerator.Default`)
- `CudaMemoryPool` - GPU memory pooling to reduce allocation overhead
- `CudaKernelCache` - Caches loaded PTX kernels
- `CudaStreamManager` - Multi-stream execution for parallelism

### Processing & Inference
- `TextPreProcessing` - Tokenization and sequence handling
- `BPETokenizer` - Byte-pair encoding tokenizer (loads vocab.json + merges.txt)
- `AudioPreProcessing` - WAV loading (8/16/24/32-bit PCM), resampling, normalization
- `MelSpectrogram` / `MelSpectrogramCuda` - STFT → mel filterbank → log compression
- `TextGenerator` - Autoregressive text generation with sampling
- `Sampler` - Greedy, temperature, top-k, top-p sampling strategies
- `KVCache` - Key-value cache for efficient transformer inference
- `SpeechSynthesizer` - TTS synthesis and vocoder integration

### Training Utilities
- `Checkpoint` - Model checkpointing with metadata (epoch, loss, LR, etc.)
- `GradientAccumulator` - Simulates larger batch sizes by accumulating gradients
- `LearningRateScheduler` - Learning rate scheduling strategies
- `Tensor` - Multi-dimensional tensor with row-major layout and shape tracking

## Development Guidelines

### Building & Running

```bash
# Build the solution
dotnet build NET_Keras.sln

# Run training
dotnet run --project NET_Keras/NET_Keras-CMD.csproj

# Run tests
dotnet test NeuralNetwork.Tests/NeuralNetwork.Tests.csproj
```

### Adding New Layers

1. Create CPU implementation in `NeuralNetwork/Layers/`
2. Inherit from `Layer` base class
3. Implement required abstract methods:
   - `Build(int[] inputShape)` - Initialize weights and biases
   - `Call(float[,] inputs)` - Forward pass (2D tensors)
   - `Call(float[,,,] inputs)` - Forward pass (4D tensors, for conv layers)
   - `Backward(float[,] gradient)` - Backward pass (2D)
   - `Backward(float[,,,] gradient)` - Backward pass (4D)
   - `GetOutputShape(int[] inputShape)` - Compute output dimensions
4. For GPU acceleration, create corresponding CUDA implementation in `Layers/Cuda/`
5. Add CUDA kernels in `NeuralNetwork/CU/` directory (`.cu` source files)

### Adding New Optimizers

1. Create implementation in `NeuralNetwork/Optimizers/`
2. Inherit from `Optimizer` base class
3. Implement `Update(float[,] weights, float[,] gradients)` and `Update4D()` methods
4. Maintain optimizer state (momentum, variance, etc.) per parameter

### Adding New Loss Functions

1. Create implementation in `NeuralNetwork/Losses/`
2. Inherit from `Loss` base class
3. Implement `Calculate(float[,] predicted, float[,] actual)` returning both loss value and gradient
4. Implement `Calculate4D()` for convolutional outputs

### Adding New Modalities

When adding support for new AI modalities (audio, speech, etc.):
1. Create preprocessing utilities in `NeuralNetwork/Processing/{Modality}/`
2. Add specialized layers if needed in `NeuralNetwork/Layers/`
3. Create inference utilities in `NeuralNetwork/Inference/{Modality}/`
4. Optionally create a high-level model class in `NeuralNetwork/Models/`
5. Add training example in `NET_Keras/Program.cs`

### CUDA Development

- PTX files are pre-compiled CUDA kernels loaded at runtime via ManagedCuda
- The build system auto-compiles `.cu` → `.ptx` if nvcc is available (see `NeuralNetwork.csproj` CompileCudaKernels target)
- Use `CudaAccelerator.Default` singleton for shared GPU context
- Use `CudaMemoryPool` to avoid repeated GPU allocations
- Follow existing patterns in `LSTMCuda.cs` or `DenseCuda.cs` for kernel loading
- Fused kernel operations (e.g., MatMul+Bias+ReLU in DenseKernel.cu) provide 1.3-2x speedups
- Test on systems with NVIDIA GPU; CUDA layers gracefully skip when no GPU is present

### Testing

- Test project: `NeuralNetwork.Tests/` (xUnit, x64 only)
- Use `GradientChecker` for numerical gradient verification (finite differences)
- Use `MatrixAssert` for matrix comparison with tolerance
- Use `TestDataBuilder` for generating test data
- CUDA tests inherit from `CudaTestFixture` which handles GPU setup/teardown

## Layer Inventory

### CPU Layers
| Layer | File | Description |
|-------|------|-------------|
| Dense | `Layers/Dense.cs` | Fully connected layer |
| Embedding | `Layers/Embedding.cs` | Token embedding lookup |
| LSTM | `Layers/LSTM.cs` | LSTM recurrent cell |
| Conv2D | `Layers/Conv2D.cs` | 2D convolution |
| MaxPool2D | `Layers/MaxPool2D.cs` | Max pooling |
| Dropout | `Layers/Dropout.cs` | Dropout regularization |
| LayerNorm | `Layers/LayerNorm.cs` | Layer normalization |
| FeedForward | `Layers/FeedForward.cs` | Two-layer MLP block |
| TransformerBlock | `Layers/TransformerBlock.cs` | Pre-LN transformer decoder block |
| PositionalEncoding | `Layers/PositionalEncoding.cs` | Sinusoidal positional encoding |
| MultiHeadAttention | `Layers/Attention/MultiHeadAttention.cs` | Multi-head attention |
| ScaledDotProductAttention | `Layers/Attention/ScaledDotProductAttention.cs` | Single attention head |
| Activations | `Layers/Activations/Activations.cs` | ReLU, Sigmoid, Tanh, Softmax, Linear |
| GELU | `Layers/Activations/GELU.cs` | GELU activation |

### CUDA Layers (GPU equivalents)
| Layer | File | Kernel |
|-------|------|--------|
| DenseCuda | `Layers/Cuda/DenseCuda.cs` | DenseKernel.ptx |
| DenseCudaOptimized | `Layers/Cuda/DenseCudaOptimized.cs` | DenseKernel.ptx (tiled) |
| LSTMCuda | `Layers/Cuda/LSTMCuda.cs` | LSTMKernel.ptx |
| GRUCuda | `Layers/Cuda/GRUCuda.cs` | GRUKernel.ptx |
| EmbeddingCuda | `Layers/Cuda/EmbeddingCuda.cs` | EmbeddingKernel.ptx |
| DropoutCuda | `Layers/Cuda/DropoutCuda.cs` | Built-in |
| LayerNormCuda | `Layers/Cuda/LayerNormCuda.cs` | LayerNormKernel.ptx |
| FeedForwardCuda | `Layers/Cuda/FeedForwardCuda.cs` | FeedForwardKernel.ptx |
| MultiHeadAttentionCuda | `Layers/Cuda/MultiHeadAttentionCuda.cs` | AttentionKernel.ptx |
| ScaledDotProductAttentionCuda | `Layers/Cuda/ScaledDotProductAttentionCuda.cs` | AttentionKernel.ptx |
| TransformerBlockCuda | `Layers/Cuda/TransformerBlockCuda.cs` | Multiple kernels |
| ActivationsCuda | `Layers/Activations/ActivationsCuda.cs` | ActivationsKernel.ptx |

## Agent Activation Rules

**IMPORTANT**: When working on tasks in this project, you MUST invoke the appropriate slash commands for guidance. Use the SlashCommand tool to expand command prompts.

### How Commands Work

Slash commands are **guidance prompts** stored in `.claude/commands/`. When invoked:
1. The command's markdown content expands into the conversation
2. Follow the instructions provided in the expanded prompt
3. Use TodoWrite to track multi-step workflows
4. Invoke follow-up commands manually using SlashCommand tool

### Automatic Activation Triggers

| When the task involves... | Invoke Command |
|---------------------------|----------------|
| Creating/implementing a new layer (Dense, Conv, Attention, etc.) | `/add-layer` |
| GPU acceleration, CUDA kernels, PTX files | `/add-cuda` |
| New optimizer (AdamW, RMSprop, LAMB, etc.) | `/add-optimizer` |
| New loss function (Focal, Huber, Triplet, etc.) | `/add-loss` |
| Audio, speech, TTS, ASR, new modality | `/add-modality` |
| Data loading, datasets, batching, preprocessing | `/data-pipeline` |
| Testing, validation, gradient checking | `/test-model` |
| Training issues: NaN, loss stuck, exploding/vanishing gradients | `/debug-training` |
| Code review, correctness check, best practices | `/review-ml` |
| Performance, speed, memory, profiling, CPU vs GPU | `/benchmark` |
| Save/load models, checkpoints, ONNX, export | `/export-model` |
| Architecture questions, design decisions, structure | `/architecture` |
| Complex multi-step tasks, workflow planning | `/coordinate` |

### Multi-Step Workflows

For complex tasks, invoke commands in sequence with TodoWrite tracking:

1. **Implementing a new layer**:
   ```
   TodoWrite: [add-layer, test-model, add-cuda, benchmark]
   SlashCommand: /add-layer → complete → /test-model → complete → /add-cuda → complete → /benchmark
   ```

2. **Adding audio support**:
   ```
   TodoWrite: [coordinate (plan), add-modality, data-pipeline, add-layer, test-model]
   SlashCommand: /coordinate → plan tasks → then invoke each command sequentially
   ```

3. **Fixing training bugs**: `/debug-training` → `/test-model` → `/review-ml`

4. **Performance optimization**: `/benchmark` → `/add-cuda` → `/test-model`

### Invocation Examples

```
User: "Add a GRU layer"
→ SlashCommand: /add-layer GRU
→ Then TodoWrite to track: [implement, test, cuda, benchmark]

User: "Training loss is NaN"
→ SlashCommand: /debug-training

User: "Make the attention layer faster"
→ SlashCommand: /benchmark attention
→ Then: /add-cuda attention

User: "Add support for audio transcription"
→ SlashCommand: /coordinate
→ Creates TodoWrite task list
→ Then invoke each command from the plan
```

## Available Slash Commands

Use these slash commands for specialized guidance. Each command expands into context-specific instructions.
Invoke using: `SlashCommand tool` with `command: "/command-name argument"`

### Core Development
- `/add-layer` - Add a new neural network layer type
- `/add-cuda` - Add CUDA GPU acceleration to a component
- `/add-optimizer` - Implement new training optimizers (AdamW, RMSprop, etc.)
- `/add-loss` - Implement new loss functions (Focal, Huber, CTC, etc.)

### Data & Modalities
- `/add-modality` - Add support for a new AI modality (audio, speech, etc.)
- `/data-pipeline` - Create data loading and preprocessing pipelines

### Quality & Testing
- `/test-model` - Test and validate model implementations
- `/debug-training` - Diagnose and fix training issues (NaN, vanishing gradients, etc.)
- `/review-ml` - Code review for correctness and best practices
- `/benchmark` - Performance benchmarking and optimization

### Infrastructure
- `/export-model` - Model serialization, checkpoints, and ONNX export
- `/architecture` - Explore and document architecture decisions
- `/coordinate` - Orchestrate multi-agent workflows and track progress

### Post-Implementation Follow-ups

After completing a task with one agent, automatically proceed to follow-up agents:

| After completing... | Then activate... |
|---------------------|------------------|
| `/add-layer` | `/test-model` to validate, then ask about `/add-cuda` |
| `/add-cuda` | `/test-model` to verify CPU/CUDA match, `/benchmark` for speedup |
| `/add-optimizer` | `/test-model` to verify convergence |
| `/add-loss` | `/test-model` to verify gradients numerically |
| `/add-modality` | `/test-model` for end-to-end validation |
| `/data-pipeline` | `/benchmark` to measure throughput |
| `/debug-training` | `/test-model` to confirm fix, `/review-ml` for root cause |
| `/export-model` | `/test-model` to verify exported model works |

### Agent Workflow Example

```
# Adding a new layer with full validation:
/architecture [LayerName]     # Understand where it fits
/add-layer [LayerName]        # Implement CPU version
/test-model [LayerName]       # Validate implementation
/add-cuda [LayerName]         # GPU acceleration
/benchmark [LayerName]        # Performance comparison
/review-ml [LayerName]        # Final code review
```

### Task Completion Checklist

Before considering any implementation task complete, ensure:
- [ ] Primary agent completed its work
- [ ] `/test-model` validated the implementation
- [ ] `/benchmark` measured performance (if applicable)
- [ ] `/review-ml` checked for issues (for significant changes)
- [ ] Documentation updated if needed
