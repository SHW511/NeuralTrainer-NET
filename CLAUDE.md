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

## Available Subagents

Use these slash commands for specialized tasks:

- `/add-layer` - Add a new neural network layer type
- `/add-cuda` - Add CUDA GPU acceleration to a component
- `/add-modality` - Add support for a new AI modality
- `/test-model` - Test and validate model implementations
- `/architecture` - Explore and document architecture decisions
