using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace NeuralNetwork.Layers.Cuda
{
    public class DenseCuda : Layer
    {
        public int OutputDim { get; private set; }
        public Func<int, int, float[,]> Init { get; private set; }
        public Func<float[,], float[,]> Activation { get; private set; }
        public float[] Biases { get; private set; }
        public bool UseBias { get; private set; }
        public float[,] InitialWeights { get; private set; }

        private float[,] inputs; // Store inputs for backpropagation
        private CudaContext context;
        private CudaDeviceVariable<float> weightsDevice;
        private CudaDeviceVariable<float> biasesDevice;

        public DenseCuda(int outputDim, Func<int, int, float[,]> init = null,
                     Func<float[,], float[,]> activation = null, float[,] weights = null,
                     bool useBias = true)
        {
            OutputDim = outputDim;
            Init = init ?? Initializers.Initializers.GlorotUniform;
            Activation = activation ?? Activations.Activations.Linear;
            InitialWeights = weights;
            UseBias = useBias;
            context = new CudaContext();
        }

        public override void Build(int[] inputShape)
        {
            if (inputShape.Length != 2)
                throw new ArgumentException("Input shape should be a 2D tensor");

            InputShape = inputShape;
            Weights = Init(inputShape[1], OutputDim);
            weightsDevice = new CudaDeviceVariable<float>(Weights.Length);
            weightsDevice.CopyToDevice(Weights);

            if (UseBias)
            {
                Biases = new float[OutputDim];
                biasesDevice = new CudaDeviceVariable<float>(Biases.Length);
                biasesDevice.CopyToDevice(Biases);
            }

            // Apply initial weights if provided
            if (InitialWeights != null)
            {
                Weights = InitialWeights;
                weightsDevice.CopyToDevice(Weights);
            }

            Built = true;
        }

        public override float[,] Call(float[,] inputs)
        {
            if (!Built)
            {
                // Determine the input shape from the inputs
                int[] inputShape = { inputs.GetLength(0), inputs.GetLength(1) };
                Build(inputShape);
            }

            this.inputs = inputs; // Store inputs for backpropagation

            var output = Dot(inputs, Weights);
            if (UseBias)
            {
                output = AddBias(output, Biases);
            }

            return Activation(output);
        }

        public override float[,] Backward(float[,] gradient)
        {
            if (InputShape == null || InputShape.Length != 2)
                throw new InvalidOperationException("InputShape must be set before calling Backward");

            int batchSize = inputs.GetLength(0);
            int inputDim = InputShape[1];
            float[,] weightGradient = new float[inputDim, OutputDim];
            float[] biasGradient = new float[OutputDim];
            float[,] inputGradient = new float[batchSize, inputDim];

            // Allocate memory on the GPU
            var inputsDevice = new CudaDeviceVariable<float>(inputs.Length);
            var gradientDevice = new CudaDeviceVariable<float>(gradient.Length);
            var weightGradientDevice = new CudaDeviceVariable<float>(weightGradient.Length);
            var biasGradientDevice = new CudaDeviceVariable<float>(biasGradient.Length);
            var inputGradientDevice = new CudaDeviceVariable<float>(inputGradient.Length);

            // Copy data to the GPU
            inputsDevice.CopyToDevice(inputs);
            gradientDevice.CopyToDevice(gradient);
            weightGradientDevice.CopyToDevice(weightGradient);
            biasGradientDevice.CopyToDevice(biasGradient);
            inputGradientDevice.CopyToDevice(inputGradient);

            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "DenseKernel.ptx");

            // Load the kernel from the .ptx file
            var kernel = context.LoadKernelPTX(path, "DenseBackward");

            // Define block and grid sizes
            dim3 blockSize = new dim3(16, 16);
            dim3 gridSize = new dim3((uint)((batchSize + blockSize.x - 1) / blockSize.x), (uint)((OutputDim + blockSize.y - 1) / blockSize.y));

            // Launch the kernel
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;
            kernel.Run(inputsDevice.DevicePointer, gradientDevice.DevicePointer, weightsDevice.DevicePointer, weightGradientDevice.DevicePointer, biasGradientDevice.DevicePointer, inputGradientDevice.DevicePointer, batchSize, inputDim, OutputDim, UseBias, 0.01f); // Example learning rate

            // Copy the result back to the CPU
            weightGradientDevice.CopyToHost(weightGradient);
            biasGradientDevice.CopyToHost(biasGradient);
            inputGradientDevice.CopyToHost(inputGradient);

            // Free GPU memory
            inputsDevice.Dispose();
            gradientDevice.Dispose();
            weightGradientDevice.Dispose();
            biasGradientDevice.Dispose();
            inputGradientDevice.Dispose();

            // Update weights and biases on the CPU
            for (int j = 0; j < OutputDim; j++)
            {
                if (UseBias)
                {
                    Biases[j] -= 0.01f * biasGradient[j]; // Example learning rate
                }
                for (int k = 0; k < inputDim; k++)
                {
                    Weights[k, j] -= 0.01f * weightGradient[k, j]; // Example learning rate
                }
            }

            weightsDevice.CopyToDevice(Weights);
            if (UseBias)
            {
                biasesDevice.CopyToDevice(Biases);
            }

            return inputGradient;
        }

        private float[,] Dot(float[,] a, float[,] b)
        {
            // Implementation of dot product using CUDA
            int aRows = a.GetLength(0);
            int aCols = a.GetLength(1);
            int bCols = b.GetLength(1);
            float[,] result = new float[aRows, bCols];

            // Allocate memory on the GPU
            var aDevice = new CudaDeviceVariable<float>(a.Length);
            var bDevice = new CudaDeviceVariable<float>(b.Length);
            var resultDevice = new CudaDeviceVariable<float>(result.Length);

            // Copy data to the GPU
            aDevice.CopyToDevice(a);
            bDevice.CopyToDevice(b);

            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "DenseKernel.ptx");

            // Compile and load the kernel
            var kernel = context.LoadKernelPTX(path, "MatMul");

            // Define block and grid sizes
            dim3 blockSize = new dim3(16, 16);
            dim3 gridSize = new dim3((uint)((bCols + blockSize.x - 1) / blockSize.x), (uint)((aRows + blockSize.y - 1) / blockSize.y));

            // Launch the kernel
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;
            kernel.Run(aDevice.DevicePointer, bDevice.DevicePointer, resultDevice.DevicePointer, aRows, aCols, bCols);

            // Copy the result back to the CPU
            resultDevice.CopyToHost(result);

            // Free GPU memory
            aDevice.Dispose();
            bDevice.Dispose();
            resultDevice.Dispose();

            return result;
        }

        private float[,] AddBias(float[,] output, float[] bias)
        {
            int rows = output.GetLength(0);
            int cols = output.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    output[i, j] += bias[j];
                }
            }
            return output;
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            if (inputShape.Length != 2)
                throw new ArgumentException("Input shape should be a 2D tensor");

            return new int[] { inputShape[0], OutputDim };
        }

        public override Dictionary<string, object> GetConfig()
        {
            var config = new Dictionary<string, object>
                                {
                                    {"outputDim", OutputDim},
                                    {"init", Init.Method.Name},
                                    {"activation", Activation.Method.Name},
                                    {"useBias", UseBias}
                                };
            var baseConfig = base.GetConfig();
            foreach (var kv in baseConfig)
            {
                config[kv.Key] = kv.Value;
            }
            return config;
        }

        public override float[,,,] Call(float[,,,] inputs)
        {
            throw new NotImplementedException();
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException();
        }
    }
}
