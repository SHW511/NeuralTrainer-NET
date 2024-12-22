using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda.VectorTypes;
using ManagedCuda;

namespace NeuralNetwork.Layers.Activations
{
    public static class ActivationsCuda
    {
        private static CudaContext context = new CudaContext();

        public static float[,] Linear(float[,] inputs)
        {
            // Create linear activation function
            return inputs;
        }

        public static float[,] ReLU(float[,] inputs)
        {
            return ReLU(inputs, float.MaxValue, 0, 0);
        }

        public static float[,] ReLU(float[,] inputs, float max_value = float.MaxValue, float threshold = 0, float negative_slope = 0)
        {
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            float[,] outputs = new float[rows, cols];

            // Allocate memory on the GPU
            var inputsDevice = new CudaDeviceVariable<float>(inputs.Length);
            var outputsDevice = new CudaDeviceVariable<float>(outputs.Length);

            // Copy data to the GPU
            inputsDevice.CopyToDevice(inputs);

            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "ActivationsKernel.ptx");

            // Load the kernel from the .ptx file
            var kernel = context.LoadKernel(path, "ReLU");

            // Define block and grid sizes
            dim3 blockSize = new dim3(16, 16);
            dim3 gridSize = new dim3((uint)((rows + blockSize.x - 1) / blockSize.x), (uint)((cols + blockSize.y - 1) / blockSize.y));

            // Launch the kernel
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;
            kernel.Run(inputsDevice.DevicePointer, outputsDevice.DevicePointer, rows, cols, max_value, threshold, negative_slope);

            // Copy the result back to the CPU
            outputsDevice.CopyToHost(outputs);

            // Free GPU memory
            inputsDevice.Dispose();
            outputsDevice.Dispose();

            return outputs;
        }

        public static float[,] Sigmoid(float[,] inputs)
        {
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            float[,] outputs = new float[rows, cols];

            // Allocate memory on the GPU
            var inputsDevice = new CudaDeviceVariable<float>(inputs.Length);
            var outputsDevice = new CudaDeviceVariable<float>(outputs.Length);

            // Copy data to the GPU
            inputsDevice.CopyToDevice(inputs);

            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "ActivationsKernel.ptx");

            // Load the kernel from the .ptx file
            var kernel = context.LoadKernel(path, "Sigmoid");

            // Define block and grid sizes
            dim3 blockSize = new dim3(16, 16);
            dim3 gridSize = new dim3((uint)((rows + blockSize.x - 1) / blockSize.x), (uint)((cols + blockSize.y - 1) / blockSize.y));

            // Launch the kernel
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;
            kernel.Run(inputsDevice.DevicePointer, outputsDevice.DevicePointer, rows, cols);

            // Copy the result back to the CPU
            outputsDevice.CopyToHost(outputs);

            // Free GPU memory
            inputsDevice.Dispose();
            outputsDevice.Dispose();

            return outputs;
        }

        public static float[,] SoftMax(float[,] inputs)
        {
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            float[,] outputs = new float[rows, cols];

            // Allocate memory on the GPU
            var inputsDevice = new CudaDeviceVariable<float>(inputs.Length);
            var outputsDevice = new CudaDeviceVariable<float>(outputs.Length);

            // Copy data to the GPU
            inputsDevice.CopyToDevice(inputs);

            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "ActivationsKernel.ptx");

            // Load the kernel from the .ptx file
            var kernel = context.LoadKernel(path, "SoftMax");

            // Define block and grid sizes
            dim3 blockSize = new dim3(16, 16);
            dim3 gridSize = new dim3((uint)((rows + blockSize.x - 1) / blockSize.x), (uint)((cols + blockSize.y - 1) / blockSize.y));

            // Launch the kernel
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;
            kernel.Run(inputsDevice.DevicePointer, outputsDevice.DevicePointer, rows, cols);

            // Copy the result back to the CPU
            outputsDevice.CopyToHost(outputs);

            // Free GPU memory
            inputsDevice.Dispose();
            outputsDevice.Dispose();

            return outputs;
        }
    }
}
