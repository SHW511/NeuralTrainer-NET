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
        private CudaDeviceVariable<float> embeddingsDevice;

        public EmbeddingCuda(int inputDim, int outputDim)
        {
            InputDim = inputDim;
            OutputDim = outputDim;
            context = new CudaContext();
        }

        public override float[,] Backward(float[,] gradient)
        {
            int samples = gradient.GetLength(0);
            int sequenceLength = gradient.GetLength(1) / OutputDim;

            // Allocate memory on the GPU
            var gradientDevice = new CudaDeviceVariable<float>(gradient.Length);
            var inputIndicesDevice = new CudaDeviceVariable<int>(inputIndices.Length);

            // Copy data to the GPU
            gradientDevice.CopyToDevice(gradient);
            inputIndicesDevice.CopyToDevice(inputIndices);

            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "EmbeddingKernel.ptx");

            // Load the kernel from the .ptx file
            var kernel = context.LoadKernel(path, "EmbeddingBackward");

            // Define block and grid sizes
            dim3 blockSize = new dim3(sequenceLength);
            dim3 gridSize = new dim3(samples);

            // Launch the kernel
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;
            kernel.Run(embeddingsDevice.DevicePointer, gradientDevice.DevicePointer, inputIndicesDevice.DevicePointer, samples, sequenceLength, InputDim, OutputDim, 0.01f); // Example learning rate

            float embResult = default;
            embeddingsDevice.CopyToHost(ref embResult);

            // Free GPU memory
            gradientDevice.Dispose();
            inputIndicesDevice.Dispose();

            // Return null as Embedding layer does not propagate gradients to previous layers
            return null;
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
            Built = true;
        }

        public override float[,] Call(float[,] inputs)
        {
            int samples = inputs.GetLength(0);
            int sequenceLength = inputs.GetLength(1);
            float[,] output = new float[samples, sequenceLength * OutputDim];
            inputIndices = new int[samples * sequenceLength];

            // Allocate memory on the GPU
            var inputsDevice = new CudaDeviceVariable<float>(inputs.Length);
            var outputDevice = new CudaDeviceVariable<float>(output.Length);

            // Copy data to the GPU
            inputsDevice.CopyToDevice(inputs);

            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "EmbeddingKernel.ptx");

            // Load the kernel from the .ptx file
            var kernel = context.LoadKernelPTX(path, "EmbeddingLookup");

            // Define block and grid sizes
            dim3 blockSize = new dim3(sequenceLength);
            dim3 gridSize = new dim3(samples);

            // Launch the kernel
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;
            kernel.Run(embeddingsDevice.DevicePointer, inputsDevice.DevicePointer, outputDevice.DevicePointer, samples, sequenceLength, OutputDim);

            // Copy the result back to the CPU
            outputDevice.CopyToHost(output);

            // Free GPU memory
            inputsDevice.Dispose();
            outputDevice.Dispose();

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
    }
}
