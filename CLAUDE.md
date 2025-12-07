# NeuralTrainer-NET

A C#/.NET implementation of a machine learning framework with CUDA GPU acceleration. "If Python can do it, I can do it too!"

## Project Vision

Build a comprehensive suite of AI training tools supporting multiple modalities:
- Text generation (implemented)
- Image classification (implemented)
- Audio transcription (planned)
- Audio generation (planned)
- Text-to-speech (planned)
- Speech-to-speech (planned)

## Architecture Overview

```
NeuralTrainer-NET/
├── NET_Keras/                    # CLI Application (entry point)
│   └── Program.cs                # Training examples and main entry
├── NeuralNetwork/                # Core ML Framework Library
│   ├── Sequential.cs             # Sequential model API
│   ├── Layer.cs                  # Abstract base layer
│   ├── Layers/                   # Layer implementations
│   │   ├── Dense.cs              # Fully connected (CPU)
│   │   ├── Embedding.cs          # Embedding layer
│   │   ├── LSTM.cs               # LSTM cell (CPU)
│   │   ├── Conv2D.cs             # Convolutional 2D
│   │   ├── MaxPool2D.cs          # Max pooling
│   │   └── Cuda/                 # GPU-accelerated layers
│   │       ├── DenseCuda.cs
│   │       ├── LSTMCuda.cs
│   │       ├── EmbeddingCuda.cs
│   │       └── GRUCuda.cs
│   ├── Optimizers/               # Training optimizers
│   │   ├── Adam.cs
│   │   └── SGD.cs
│   ├── Losses/                   # Loss functions
│   │   ├── MeanSquaredError.cs
│   │   └── CategoricalCrossentropy.cs
│   ├── CU/                       # CUDA kernels (.cu and .ptx)
│   ├── Processing/Text/          # Text preprocessing
│   └── Inference/                # Model inference utilities
```

## Technology Stack

- **Language**: C# (.NET 9.0)
- **GPU**: CUDA via ManagedCuda v10.0.0
- **Build**: MSBuild / Visual Studio 2022+
- **Platform**: x64 (required for CUDA)

## Development Guidelines

### Adding New Layers

1. Create CPU implementation in `NeuralNetwork/Layers/`
2. Inherit from `Layer` base class
3. Implement `Call()`, `Backward()`, and `Build()` methods
4. For GPU acceleration, create corresponding CUDA implementation in `Layers/Cuda/`
5. Add CUDA kernels in `NeuralNetwork/CU/` directory

### Adding New Modalities

When adding support for new AI modalities (audio, speech, etc.):
1. Create preprocessing utilities in `NeuralNetwork/Processing/{Modality}/`
2. Add specialized layers if needed (e.g., attention, transformers)
3. Create inference utilities in `NeuralNetwork/Inference/`
4. Add training example in `NET_Keras/Program.cs`

### CUDA Development

- PTX files are pre-compiled CUDA kernels loaded at runtime
- Use `ManagedCuda.CudaContext` for GPU memory management
- Follow existing patterns in `LSTMCuda.cs` for kernel loading
- Test thoroughly on systems with NVIDIA GPU

### Building & Running

```bash
# Build the solution
dotnet build NET_Keras.sln

# Run training
dotnet run --project NET_Keras/NET_Keras-CMD.csproj
```

## Key Classes

- `Sequential` - Main model class, Keras-like API
- `Layer` - Abstract base for all layers
- `Optimizer` - Abstract base for optimizers
- `Loss` - Abstract base for loss functions
- `TextPreProcessing` - Tokenization and sequence handling
- `TextGenerator` - Inference for text models

## Agent Activation Rules

**IMPORTANT**: When working on tasks in this project, you MUST automatically activate the appropriate agents based on task context. Do not wait for the user to explicitly request an agent.

### Automatic Activation Triggers

| When the task involves... | Activate Agent |
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

### Multi-Agent Activation

For complex tasks, activate multiple agents in sequence:

1. **Implementing a new layer**: `/add-layer` → `/test-model` → `/add-cuda` → `/benchmark`
2. **Adding audio support**: `/coordinate` → `/add-modality` → `/data-pipeline` → `/add-layer`
3. **Fixing training bugs**: `/debug-training` → `/test-model` → `/review-ml`
4. **Performance optimization**: `/benchmark` → `/add-cuda` → `/test-model`

### Activation Examples

```
User: "Add a GRU layer"
→ Activate: /add-layer GRU

User: "Training loss is NaN"
→ Activate: /debug-training

User: "Make the attention layer faster"
→ Activate: /benchmark attention, then /add-cuda attention

User: "Add support for audio transcription"
→ Activate: /coordinate, then follow multi-step workflow
```

## Available Agents

Use these slash commands for specialized tasks. Agents can reference each other and work together.

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
