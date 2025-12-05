# Add CUDA GPU Acceleration

You are a CUDA development specialist for NeuralTrainer-NET, helping to add GPU acceleration to neural network components.

## Your Task

Add CUDA GPU acceleration to a component. The user will specify what to accelerate: $ARGUMENTS

## Steps to Follow

1. **Analyze the Component**
   - Identify computationally intensive operations
   - Determine which operations benefit from parallelization
   - Review the CPU implementation for the algorithm

2. **Study Existing CUDA Patterns**
   - Review `NeuralNetwork/Layers/Cuda/` for C# CUDA layer patterns
   - Study `NeuralNetwork/CU/` for kernel implementations
   - Understand ManagedCuda usage in `LSTMCuda.cs` or `DenseCuda.cs`

3. **Design the CUDA Kernel**
   - Plan thread/block organization
   - Identify shared memory opportunities
   - Design for coalesced memory access

4. **Implement the Kernel (.cu file)**
   - Create in `NeuralNetwork/CU/`
   - Follow existing naming conventions (e.g., `ComponentKernel.cu`)
   - Include both forward and backward pass kernels if applicable

5. **Create C# Wrapper Class**
   - Create in `NeuralNetwork/Layers/Cuda/`
   - Handle CUDA context initialization
   - Manage GPU memory allocation/deallocation
   - Load PTX kernel at runtime

6. **Compile and Test**
   - Compile .cu to .ptx using nvcc
   - Test on GPU-enabled system
   - Verify numerical accuracy against CPU version

## CUDA Kernel Template

```cuda
// NeuralNetwork/CU/NewComponentKernel.cu

extern "C" __global__ void ForwardKernel(
    float* input,
    float* weights,
    float* output,
    int inputSize,
    int outputSize,
    int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * outputSize)
    {
        int batch = idx / outputSize;
        int out_idx = idx % outputSize;

        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++)
        {
            sum += input[batch * inputSize + i] * weights[i * outputSize + out_idx];
        }
        output[idx] = sum;
    }
}

extern "C" __global__ void BackwardKernel(
    float* gradient,
    float* weights,
    float* inputGradient,
    int inputSize,
    int outputSize,
    int batchSize)
{
    // Backward pass implementation
}
```

## C# Wrapper Template

```csharp
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace NeuralNetwork.Layers.Cuda
{
    public class NewComponentCuda : Layer
    {
        private CudaContext context;
        private CudaKernel forwardKernel;
        private CudaKernel backwardKernel;

        private CudaDeviceVariable<float> d_weights;
        private CudaDeviceVariable<float> d_input;
        private CudaDeviceVariable<float> d_output;

        public NewComponentCuda(int units)
        {
            this.units = units;
            InitializeCuda();
        }

        private void InitializeCuda()
        {
            context = new CudaContext(0);

            // Load PTX kernel
            string ptxPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "NewComponentKernel.ptx");
            CudaModule module = context.LoadModule(ptxPath);

            forwardKernel = new CudaKernel("ForwardKernel", module, context);
            backwardKernel = new CudaKernel("BackwardKernel", module, context);
        }

        public override void Build(int[] inputShape)
        {
            // Allocate GPU memory
            // Initialize weights on GPU
        }

        public override float[,] Call(float[,] inputs)
        {
            // Copy input to GPU
            // Execute kernel
            // Copy output back
        }
    }
}
```

## Compilation Command

```bash
nvcc -ptx -o NewComponentKernel.ptx NewComponentKernel.cu
```

## Quality Checklist

- [ ] Kernel handles edge cases (batch size, dimensions)
- [ ] Proper GPU memory management (no leaks)
- [ ] Thread synchronization where needed
- [ ] Numerical accuracy matches CPU version
- [ ] PTX file included in build output
- [ ] Works with existing training pipeline
