using System;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Layers.Cuda
{
    /// <summary>
    /// CUDA-accelerated Layer Normalization.
    /// Normalizes across the last dimension (features) for each sample.
    /// </summary>
    public class LayerNormCuda : Layer, IDisposable
    {
        private readonly int _normalizedShape;
        private readonly float _epsilon;

        private CudaContext _context;
        private bool _contextOwned;
        private string _kernelPath;

        // Parameters on device
        private CudaDeviceVariable<float> _gammaDevice;
        private CudaDeviceVariable<float> _betaDevice;

        // Host copies
        private float[] _gamma;
        private float[] _beta;

        // Gradient accumulators
        private float[] _gammaGrad;
        private float[] _betaGrad;

        // Cached values for backward pass (on device)
        private CudaDeviceVariable<float> _meanDevice;
        private CudaDeviceVariable<float> _varianceDevice;
        private CudaDeviceVariable<float> _normalizedDevice;
        private CudaDeviceVariable<float> _lastInputDevice;

        // Track allocated sizes
        private int _allocatedBatchSize;
        private int _allocatedFeatures;

        public float[] Gamma => _gamma;
        public float[] Beta => _beta;
        public float[] GammaGrad => _gammaGrad;
        public float[] BetaGrad => _betaGrad;

        /// <summary>
        /// Create a CUDA-accelerated LayerNorm layer.
        /// </summary>
        /// <param name="normalizedShape">Size of the last dimension to normalize over.</param>
        /// <param name="epsilon">Small constant for numerical stability.</param>
        /// <param name="context">Optional shared CUDA context.</param>
        public LayerNormCuda(int normalizedShape, float epsilon = 1e-5f, CudaContext context = null)
        {
            _normalizedShape = normalizedShape;
            _epsilon = epsilon;
            OutputDim = normalizedShape;

            if (context != null)
            {
                _context = context;
                _contextOwned = false;
            }
            else
            {
                _context = new CudaContext();
                _contextOwned = true;
            }

            _kernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "LayerNormKernel.ptx");
        }

        public override void Build(int[] inputShape)
        {
            if (Built) return;

            InputShape = inputShape;
            InputDim = inputShape[inputShape.Length - 1];

            if (InputDim != _normalizedShape)
                throw new ArgumentException($"Input last dimension {InputDim} doesn't match normalized shape {_normalizedShape}");

            // Initialize gamma to 1 and beta to 0
            _gamma = new float[_normalizedShape];
            _beta = new float[_normalizedShape];
            _gammaGrad = new float[_normalizedShape];
            _betaGrad = new float[_normalizedShape];

            for (int i = 0; i < _normalizedShape; i++)
            {
                _gamma[i] = 1.0f;
                _beta[i] = 0.0f;
            }

            // Allocate device memory for parameters
            _gammaDevice = new CudaDeviceVariable<float>(_normalizedShape);
            _betaDevice = new CudaDeviceVariable<float>(_normalizedShape);

            _gammaDevice.CopyToDevice(_gamma);
            _betaDevice.CopyToDevice(_beta);

            Built = true;
        }

        private void EnsureCacheAllocated(int batchSize, int features)
        {
            if (_allocatedBatchSize >= batchSize && _allocatedFeatures >= features)
                return;

            // Dispose old allocations
            _meanDevice?.Dispose();
            _varianceDevice?.Dispose();
            _normalizedDevice?.Dispose();
            _lastInputDevice?.Dispose();

            // Allocate new
            _meanDevice = new CudaDeviceVariable<float>(batchSize);
            _varianceDevice = new CudaDeviceVariable<float>(batchSize);
            _normalizedDevice = new CudaDeviceVariable<float>(batchSize * features);
            _lastInputDevice = new CudaDeviceVariable<float>(batchSize * features);

            _allocatedBatchSize = batchSize;
            _allocatedFeatures = features;
        }

        /// <summary>
        /// Forward pass for 2D input [batch, features].
        /// </summary>
        public override float[,] Call(float[,] inputs)
        {
            int batchSize = inputs.GetLength(0);
            int features = inputs.GetLength(1);

            if (!Built)
                Build(new[] { batchSize, features });

            EnsureCacheAllocated(batchSize, features);

            int size = batchSize * features;

            using var inputDevice = new CudaDeviceVariable<float>(size);
            using var outputDevice = new CudaDeviceVariable<float>(size);

            // Flatten and copy input
            float[] inputFlat = Flatten2D(inputs);
            inputDevice.CopyToDevice(inputFlat);
            _lastInputDevice.CopyToDevice(inputFlat);

            // Run kernel
            var kernel = _context.LoadKernelPTX(_kernelPath, "LayerNormForward");

            int blockSize = Math.Min(256, features);
            blockSize = (int)Math.Pow(2, Math.Ceiling(Math.Log(blockSize) / Math.Log(2)));
            blockSize = Math.Max(32, Math.Min(256, blockSize));

            kernel.GridDimensions = new dim3((uint)batchSize, 1, 1);
            kernel.BlockDimensions = new dim3((uint)blockSize, 1, 1);
            kernel.DynamicSharedMemory = (uint)(blockSize * sizeof(float));

            kernel.Run(
                inputDevice.DevicePointer,
                _gammaDevice.DevicePointer,
                _betaDevice.DevicePointer,
                outputDevice.DevicePointer,
                _meanDevice.DevicePointer,
                _varianceDevice.DevicePointer,
                _normalizedDevice.DevicePointer,
                batchSize,
                features,
                _epsilon);

            // Copy output back
            float[] outputFlat = new float[size];
            outputDevice.CopyToHost(outputFlat);

            return Unflatten2D(outputFlat, batchSize, features);
        }

        /// <summary>
        /// Forward pass for 3D input [batch, seqLen, features] using Tensor API.
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            if (input.Rank == 2)
            {
                float[,] input2D = input.ToArray2D();
                float[,] output2D = Call(input2D);
                return Tensor.FromArray(output2D);
            }

            if (input.Rank != 3)
                throw new ArgumentException($"LayerNormCuda expects rank-2 or rank-3 input, got {input.Rank}");

            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int features = input.Shape[2];
            int totalRows = batch * seqLen;

            if (!Built)
                Build(new[] { batch, seqLen, features });

            EnsureCacheAllocated(totalRows, features);

            int size = totalRows * features;

            using var inputDevice = new CudaDeviceVariable<float>(size);
            using var outputDevice = new CudaDeviceVariable<float>(size);

            inputDevice.CopyToDevice(input.Data);
            _lastInputDevice.CopyToDevice(input.Data);

            var kernel = _context.LoadKernelPTX(_kernelPath, "LayerNormForward3D");

            int blockSize = Math.Min(256, features);
            blockSize = (int)Math.Pow(2, Math.Ceiling(Math.Log(blockSize) / Math.Log(2)));
            blockSize = Math.Max(32, Math.Min(256, blockSize));

            kernel.GridDimensions = new dim3((uint)totalRows, 1, 1);
            kernel.BlockDimensions = new dim3((uint)blockSize, 1, 1);
            kernel.DynamicSharedMemory = (uint)(blockSize * sizeof(float));

            kernel.Run(
                inputDevice.DevicePointer,
                _gammaDevice.DevicePointer,
                _betaDevice.DevicePointer,
                outputDevice.DevicePointer,
                _meanDevice.DevicePointer,
                _varianceDevice.DevicePointer,
                _normalizedDevice.DevicePointer,
                totalRows,
                features,
                _epsilon);

            float[] outputData = new float[size];
            outputDevice.CopyToHost(outputData);

            return new Tensor(outputData, input.Shape);
        }

        /// <summary>
        /// Backward pass for 2D gradient.
        /// </summary>
        public override float[,] Backward(float[,] gradient)
        {
            int batchSize = gradient.GetLength(0);
            int features = gradient.GetLength(1);
            int size = batchSize * features;

            // Reset gradient accumulators
            Array.Clear(_gammaGrad, 0, _gammaGrad.Length);
            Array.Clear(_betaGrad, 0, _betaGrad.Length);

            using var gradOutputDevice = new CudaDeviceVariable<float>(size);
            using var gradInputDevice = new CudaDeviceVariable<float>(size);
            using var gradGammaDevice = new CudaDeviceVariable<float>(features);
            using var gradBetaDevice = new CudaDeviceVariable<float>(features);

            float[] gradFlat = Flatten2D(gradient);
            gradOutputDevice.CopyToDevice(gradFlat);
            gradGammaDevice.CopyToDevice(_gammaGrad);
            gradBetaDevice.CopyToDevice(_betaGrad);

            var kernel = _context.LoadKernelPTX(_kernelPath, "LayerNormBackward");

            int blockSize = Math.Min(256, features);
            blockSize = (int)Math.Pow(2, Math.Ceiling(Math.Log(blockSize) / Math.Log(2)));
            blockSize = Math.Max(32, Math.Min(256, blockSize));

            kernel.GridDimensions = new dim3((uint)batchSize, 1, 1);
            kernel.BlockDimensions = new dim3((uint)blockSize, 1, 1);
            kernel.DynamicSharedMemory = (uint)(3 * blockSize * sizeof(float));

            kernel.Run(
                gradOutputDevice.DevicePointer,
                _lastInputDevice.DevicePointer,
                _gammaDevice.DevicePointer,
                _meanDevice.DevicePointer,
                _varianceDevice.DevicePointer,
                _normalizedDevice.DevicePointer,
                gradInputDevice.DevicePointer,
                gradGammaDevice.DevicePointer,
                gradBetaDevice.DevicePointer,
                batchSize,
                features,
                _epsilon);

            // Copy results back
            float[] gradInputFlat = new float[size];
            gradInputDevice.CopyToHost(gradInputFlat);
            gradGammaDevice.CopyToHost(_gammaGrad);
            gradBetaDevice.CopyToHost(_betaGrad);

            return Unflatten2D(gradInputFlat, batchSize, features);
        }

        /// <summary>
        /// Backward pass for 3D gradient using Tensor API.
        /// </summary>
        public Tensor Backward(Tensor gradOutput)
        {
            if (gradOutput.Rank == 2)
            {
                float[,] grad2D = gradOutput.ToArray2D();
                float[,] dInput2D = Backward(grad2D);
                return Tensor.FromArray(dInput2D);
            }

            if (gradOutput.Rank != 3)
                throw new ArgumentException($"LayerNormCuda backward expects rank-2 or rank-3 gradient, got {gradOutput.Rank}");

            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];
            int features = gradOutput.Shape[2];
            int totalRows = batch * seqLen;
            int size = totalRows * features;

            // Reset gradient accumulators
            Array.Clear(_gammaGrad, 0, _gammaGrad.Length);
            Array.Clear(_betaGrad, 0, _betaGrad.Length);

            using var gradOutputDevice = new CudaDeviceVariable<float>(size);
            using var gradInputDevice = new CudaDeviceVariable<float>(size);
            using var gradGammaDevice = new CudaDeviceVariable<float>(features);
            using var gradBetaDevice = new CudaDeviceVariable<float>(features);

            gradOutputDevice.CopyToDevice(gradOutput.Data);
            gradGammaDevice.CopyToDevice(_gammaGrad);
            gradBetaDevice.CopyToDevice(_betaGrad);

            var kernel = _context.LoadKernelPTX(_kernelPath, "LayerNormBackward3D");

            int blockSize = Math.Min(256, features);
            blockSize = (int)Math.Pow(2, Math.Ceiling(Math.Log(blockSize) / Math.Log(2)));
            blockSize = Math.Max(32, Math.Min(256, blockSize));

            kernel.GridDimensions = new dim3((uint)totalRows, 1, 1);
            kernel.BlockDimensions = new dim3((uint)blockSize, 1, 1);
            kernel.DynamicSharedMemory = (uint)(3 * blockSize * sizeof(float));

            kernel.Run(
                gradOutputDevice.DevicePointer,
                _lastInputDevice.DevicePointer,
                _gammaDevice.DevicePointer,
                _meanDevice.DevicePointer,
                _varianceDevice.DevicePointer,
                _normalizedDevice.DevicePointer,
                gradInputDevice.DevicePointer,
                gradGammaDevice.DevicePointer,
                gradBetaDevice.DevicePointer,
                totalRows,
                features,
                _epsilon);

            float[] gradInputData = new float[size];
            gradInputDevice.CopyToHost(gradInputData);
            gradGammaDevice.CopyToHost(_gammaGrad);
            gradBetaDevice.CopyToHost(_betaGrad);

            return new Tensor(gradInputData, gradOutput.Shape);
        }

        /// <summary>
        /// Update parameters on device after host modification.
        /// </summary>
        public void SyncParametersToDevice()
        {
            _gammaDevice?.CopyToDevice(_gamma);
            _betaDevice?.CopyToDevice(_beta);
        }

        /// <summary>
        /// Update parameters using gradients.
        /// </summary>
        public void UpdateParameters(float learningRate)
        {
            for (int i = 0; i < _normalizedShape; i++)
            {
                _gamma[i] -= learningRate * _gammaGrad[i];
                _beta[i] -= learningRate * _betaGrad[i];
            }
            SyncParametersToDevice();
        }

        private static float[] Flatten2D(float[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            float[] flat = new float[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    flat[i * cols + j] = array[i, j];
                }
            }
            return flat;
        }

        private static float[,] Unflatten2D(float[] flat, int rows, int cols)
        {
            float[,] array = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    array[i, j] = flat[i * cols + j];
                }
            }
            return array;
        }

        public override float[,,,] Call(float[,,,] inputs)
        {
            throw new NotImplementedException("LayerNormCuda does not support 4D tensors directly");
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException("LayerNormCuda does not support 4D tensors directly");
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return inputShape;
        }

        public void Dispose()
        {
            _gammaDevice?.Dispose();
            _betaDevice?.Dispose();
            _meanDevice?.Dispose();
            _varianceDevice?.Dispose();
            _normalizedDevice?.Dispose();
            _lastInputDevice?.Dispose();

            if (_contextOwned)
            {
                _context?.Dispose();
            }
        }
    }
}
