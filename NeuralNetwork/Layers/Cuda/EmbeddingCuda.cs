using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.VectorTypes;

namespace NeuralNetwork.Layers.Cuda
{
    public class EmbeddingCuda : Layer
    {
        private float[,] embeddings;
        private int[] inputIndices; // Store input indices for backpropagation
        private CudaContext context;
        private bool _contextOwned;
        private CudaDeviceVariable<float> embeddingsDevice;
        private bool _disposed;

        // Cached kernel path and kernels
        private string _kernelPath;
        private CudaKernel _lookupKernel;
        private CudaKernel _backwardKernel;

        public EmbeddingCuda(int inputDim, int outputDim, CudaContext context = null)
        {
            InputDim = inputDim;
            OutputDim = outputDim;

            if (context != null)
            {
                this.context = context;
                _contextOwned = false;
            }
            else
            {
                this.context = new CudaContext();
                _contextOwned = true;
            }

            _kernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "EmbeddingKernel.ptx");
        }

        public override float[,] Backward(float[,] gradient)
        {
            int samples = gradient.GetLength(0);
            int sequenceLength = gradient.GetLength(1) / OutputDim;

            using var gradientDevice = new CudaDeviceVariable<float>(gradient.Length);
            using var inputIndicesDevice = new CudaDeviceVariable<int>(inputIndices.Length);

            gradientDevice.CopyToDevice(gradient);
            inputIndicesDevice.CopyToDevice(inputIndices);

            // Use cached backward kernel (falls back to loading if not yet cached)
            if (_backwardKernel == null)
                _backwardKernel = context.LoadKernel(_kernelPath, "EmbeddingBackward");

            dim3 blockSize = new dim3(sequenceLength);
            dim3 gridSize = new dim3(samples);

            _backwardKernel.GridDimensions = gridSize;
            _backwardKernel.BlockDimensions = blockSize;
            _backwardKernel.Run(embeddingsDevice.DevicePointer, gradientDevice.DevicePointer, inputIndicesDevice.DevicePointer, samples, sequenceLength, InputDim, OutputDim, 0.01f); // Example learning rate

            float[,] embResult = new float[gradient.Length, inputIndices.Length];
            embeddingsDevice.CopyToHost(embResult);

            return embResult;
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException();
        }

        public override void Build(int[] inputShape)
        {
            embeddings = Initializers.Initializers.GlorotUniform(InputDim, OutputDim);
            embeddingsDevice = new CudaDeviceVariable<float>(embeddings.Length);
            embeddingsDevice.CopyToDevice(embeddings);

            // Pre-load and cache kernels
            _lookupKernel = context.LoadKernelPTX(_kernelPath, "EmbeddingLookup");
            _backwardKernel = context.LoadKernel(_kernelPath, "EmbeddingBackward");

            Built = true;
        }

        public override float[,] Call(float[,] inputs)
        {
            int samples = inputs.GetLength(0);
            int sequenceLength = inputs.GetLength(1);
            float[,] output = new float[samples, sequenceLength * OutputDim];
            inputIndices = new int[samples * sequenceLength];

            // Allocate memory on the GPU with using statements to prevent leaks on exceptions
            using var inputsDevice = new CudaDeviceVariable<float>(inputs.Length);
            using var outputDevice = new CudaDeviceVariable<float>(output.Length);

            // Copy data to the GPU
            inputsDevice.CopyToDevice(inputs);

            // Use cached kernel
            var kernel = _lookupKernel;

            // Define block and grid sizes
            dim3 blockSize = new dim3(128); //dim3(sequenceLength);
            dim3 gridSize = new dim3(samples);

            // Launch the kernel
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;
            kernel.Run(embeddingsDevice.DevicePointer, inputsDevice.DevicePointer, outputDevice.DevicePointer, samples, sequenceLength, OutputDim);

            // Copy the result back to the CPU
            outputDevice.CopyToHost(output);

            return output;
        }

        public override float[,,,] Call(float[,,,] inputs)
        {
            throw new NotImplementedException();
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return new int[] { inputShape[0], inputShape[1] * OutputDim };
        }

        public override void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            embeddingsDevice?.Dispose();

            if (_contextOwned)
            {
                context?.Dispose();
            }

            GC.SuppressFinalize(this);
        }

        ~EmbeddingCuda() => Dispose();
    }
}
