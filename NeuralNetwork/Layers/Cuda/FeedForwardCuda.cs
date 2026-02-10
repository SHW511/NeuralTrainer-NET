using System;
using System.IO;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using NeuralNetwork.Tensors;
using NeuralNetwork.Initializers;

namespace NeuralNetwork.Layers.Cuda
{
    /// <summary>
    /// CUDA-accelerated Position-wise Feed-Forward Network.
    /// FFN(x) = Linear2(GELU(Linear1(x)))
    /// </summary>
    public class FeedForwardCuda : Layer
    {
        private readonly int _modelDim;
        private readonly int _ffDim;
        private readonly float _dropout;

        private CudaContext _context;
        private bool _contextOwned;
        private string _kernelPath;
        private string _denseKernelPath;

        // Weights on device
        private CudaDeviceVariable<float> _w1Device;
        private CudaDeviceVariable<float> _b1Device;
        private CudaDeviceVariable<float> _w2Device;
        private CudaDeviceVariable<float> _b2Device;

        // Host copies
        private float[,] _w1;
        private float[] _b1;
        private float[,] _w2;
        private float[] _b2;

        // Gradient accumulators
        private float[,] _w1Grad;
        private float[] _b1Grad;
        private float[,] _w2Grad;
        private float[] _b2Grad;

        // Cached values for backward pass
        private Tensor _lastInput;
        private CudaDeviceVariable<float> _hiddenDevice;
        private CudaDeviceVariable<float> _preGeluDevice;
        private int _allocatedBatchSize;

        private bool _training;

        public bool Training
        {
            get => _training;
            set => _training = value;
        }

        // Expose weights for optimizer
        public float[,] W1 => _w1;
        public float[,] W2 => _w2;
        public float[] B1 => _b1;
        public float[] B2 => _b2;
        public float[,] W1Grad => _w1Grad;
        public float[,] W2Grad => _w2Grad;
        public float[] B1Grad => _b1Grad;
        public float[] B2Grad => _b2Grad;

        /// <summary>
        /// Create a CUDA-accelerated Feed-Forward Network layer.
        /// </summary>
        /// <param name="modelDim">Model dimension (d_model).</param>
        /// <param name="ffDim">Feed-forward dimension (d_ff). Typically 4 * d_model.</param>
        /// <param name="dropout">Dropout rate.</param>
        /// <param name="context">Optional shared CUDA context.</param>
        public FeedForwardCuda(int modelDim, int ffDim, float dropout = 0.1f, CudaContext context = null)
        {
            _modelDim = modelDim;
            _ffDim = ffDim;
            _dropout = dropout;
            _training = true;
            OutputDim = modelDim;

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

            _kernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "FeedForwardKernel.ptx");
            _denseKernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "DenseKernel.ptx");
        }

        public override void Build(int[] inputShape)
        {
            if (Built) return;

            InputShape = inputShape;
            InputDim = inputShape[inputShape.Length - 1];

            if (InputDim != _modelDim)
                throw new ArgumentException($"Input dimension {InputDim} doesn't match model dimension {_modelDim}");

            // Initialize weights using Glorot uniform
            _w1 = Initializers.Initializers.GlorotUniform(_modelDim, _ffDim);
            _w2 = Initializers.Initializers.GlorotUniform(_ffDim, _modelDim);

            // Initialize biases to zero
            _b1 = new float[_ffDim];
            _b2 = new float[_modelDim];

            // Initialize gradient accumulators
            _w1Grad = new float[_modelDim, _ffDim];
            _b1Grad = new float[_ffDim];
            _w2Grad = new float[_ffDim, _modelDim];
            _b2Grad = new float[_modelDim];

            // Allocate device memory for weights
            _w1Device = new CudaDeviceVariable<float>(_modelDim * _ffDim);
            _b1Device = new CudaDeviceVariable<float>(_ffDim);
            _w2Device = new CudaDeviceVariable<float>(_ffDim * _modelDim);
            _b2Device = new CudaDeviceVariable<float>(_modelDim);

            // Copy weights to device
            CopyWeightsToDevice();

            Built = true;
        }

        private void CopyWeightsToDevice()
        {
            _w1Device.CopyToDevice(Flatten(_w1));
            _b1Device.CopyToDevice(_b1);
            _w2Device.CopyToDevice(Flatten(_w2));
            _b2Device.CopyToDevice(_b2);
        }

        private void EnsureCacheAllocated(int batchSize)
        {
            if (_allocatedBatchSize >= batchSize)
                return;

            _hiddenDevice?.Dispose();
            _preGeluDevice?.Dispose();

            _hiddenDevice = new CudaDeviceVariable<float>(batchSize * _ffDim);
            _preGeluDevice = new CudaDeviceVariable<float>(batchSize * _ffDim);

            _allocatedBatchSize = batchSize;
        }

        private static float[] Flatten(float[,] array)
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

        /// <summary>
        /// Forward pass using Tensor API with CUDA acceleration.
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            if (input.Rank != 3)
                throw new ArgumentException($"FeedForward expects rank-3 input, got {input.Rank}");

            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int totalRows = batch * seqLen;

            if (!Built)
                Build(new[] { batch, seqLen, input.Shape[2] });

            _lastInput = input;
            EnsureCacheAllocated(totalRows);

            int inputSize = totalRows * _modelDim;
            int hiddenSize = totalRows * _ffDim;
            int outputSize = totalRows * _modelDim;

            using var inputDevice = new CudaDeviceVariable<float>(inputSize);
            using var hiddenDevice = new CudaDeviceVariable<float>(hiddenSize);
            using var outputDevice = new CudaDeviceVariable<float>(outputSize);

            inputDevice.CopyToDevice(input.Data);

            // Step 1: First linear layer - input @ w1 + b1
            var linearKernel = _context.LoadKernelPTX(_kernelPath, "LinearForward");
            dim3 blockSize = new dim3(16, 16);
            dim3 gridSize = new dim3(
                (uint)((_ffDim + blockSize.x - 1) / blockSize.x),
                (uint)((totalRows + blockSize.y - 1) / blockSize.y));

            linearKernel.GridDimensions = gridSize;
            linearKernel.BlockDimensions = blockSize;
            linearKernel.Run(
                inputDevice.DevicePointer,
                _w1Device.DevicePointer,
                _b1Device.DevicePointer,
                _preGeluDevice.DevicePointer,
                totalRows,
                _modelDim,
                _ffDim);

            // Step 2: GELU activation
            var geluKernel = _context.LoadKernelPTX(_kernelPath, "GELU");
            int geluBlockSize = 256;
            int geluGridSize = (hiddenSize + geluBlockSize - 1) / geluBlockSize;

            geluKernel.GridDimensions = new dim3((uint)geluGridSize, 1, 1);
            geluKernel.BlockDimensions = new dim3((uint)geluBlockSize, 1, 1);
            geluKernel.Run(
                _preGeluDevice.DevicePointer,
                _hiddenDevice.DevicePointer,
                hiddenSize);

            // Step 3: Dropout (if training)
            if (_training && _dropout > 0f)
            {
                ApplyDropoutDevice(_hiddenDevice, totalRows, _ffDim);
            }

            // Step 4: Second linear layer - hidden @ w2 + b2
            dim3 gridSize2 = new dim3(
                (uint)((_modelDim + blockSize.x - 1) / blockSize.x),
                (uint)((totalRows + blockSize.y - 1) / blockSize.y));

            linearKernel.GridDimensions = gridSize2;
            linearKernel.BlockDimensions = blockSize;
            linearKernel.Run(
                _hiddenDevice.DevicePointer,
                _w2Device.DevicePointer,
                _b2Device.DevicePointer,
                outputDevice.DevicePointer,
                totalRows,
                _ffDim,
                _modelDim);

            // Copy result back
            float[] outputData = new float[outputSize];
            outputDevice.CopyToHost(outputData);

            return new Tensor(outputData, input.Shape);
        }

        /// <summary>
        /// Backward pass using Tensor API.
        /// </summary>
        public Tensor Backward(Tensor gradOutput)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];
            int totalRows = batch * seqLen;

            // Reset gradient accumulators
            ClearGradients();

            int inputSize = totalRows * _modelDim;
            int hiddenSize = totalRows * _ffDim;

            using var gradOutputDevice = new CudaDeviceVariable<float>(inputSize);
            using var gradHiddenDevice = new CudaDeviceVariable<float>(hiddenSize);
            using var gradPreGeluDevice = new CudaDeviceVariable<float>(hiddenSize);
            using var gradInputDevice = new CudaDeviceVariable<float>(inputSize);
            using var gradW1Device = new CudaDeviceVariable<float>(_modelDim * _ffDim);
            using var gradB1Device = new CudaDeviceVariable<float>(_ffDim);
            using var gradW2Device = new CudaDeviceVariable<float>(_ffDim * _modelDim);
            using var gradB2Device = new CudaDeviceVariable<float>(_modelDim);
            using var inputDevice = new CudaDeviceVariable<float>(inputSize);

            gradOutputDevice.CopyToDevice(gradOutput.Data);
            inputDevice.CopyToDevice(_lastInput.Data);

            // Initialize gradient accumulators to zero
            gradW1Device.Memset(0);
            gradB1Device.Memset(0);
            gradW2Device.Memset(0);
            gradB2Device.Memset(0);

            dim3 blockSize = new dim3(16, 16);

            // Step 1: Backward through second linear layer
            // Gradient w.r.t. hidden
            var linearBackwardInputKernel = _context.LoadKernelPTX(_kernelPath, "LinearBackwardInput");
            dim3 gradHiddenGridSize = new dim3(
                (uint)((_ffDim + blockSize.x - 1) / blockSize.x),
                (uint)((totalRows + blockSize.y - 1) / blockSize.y));

            linearBackwardInputKernel.GridDimensions = gradHiddenGridSize;
            linearBackwardInputKernel.BlockDimensions = blockSize;
            linearBackwardInputKernel.Run(
                gradOutputDevice.DevicePointer,
                _w2Device.DevicePointer,
                gradHiddenDevice.DevicePointer,
                totalRows,
                _ffDim,
                _modelDim);

            // Gradient w.r.t. w2 and b2
            var linearBackwardKernel = _context.LoadKernelPTX(_kernelPath, "LinearBackward");
            dim3 gradW2GridSize = new dim3(
                (uint)((_modelDim + blockSize.x - 1) / blockSize.x),
                (uint)((_ffDim + blockSize.y - 1) / blockSize.y));

            linearBackwardKernel.GridDimensions = gradW2GridSize;
            linearBackwardKernel.BlockDimensions = blockSize;
            linearBackwardKernel.Run(
                gradOutputDevice.DevicePointer,
                _hiddenDevice.DevicePointer,
                _w2Device.DevicePointer,
                gradHiddenDevice.DevicePointer,  // Not used, but needed for signature
                gradW2Device.DevicePointer,
                gradB2Device.DevicePointer,
                totalRows,
                _ffDim,
                _modelDim);

            // Step 2: Backward through GELU
            var geluBackwardKernel = _context.LoadKernelPTX(_kernelPath, "GELUBackward");
            int geluBlockSize = 256;
            int geluGridSize = (hiddenSize + geluBlockSize - 1) / geluBlockSize;

            geluBackwardKernel.GridDimensions = new dim3((uint)geluGridSize, 1, 1);
            geluBackwardKernel.BlockDimensions = new dim3((uint)geluBlockSize, 1, 1);
            geluBackwardKernel.Run(
                _preGeluDevice.DevicePointer,
                gradHiddenDevice.DevicePointer,
                gradPreGeluDevice.DevicePointer,
                hiddenSize);

            // Step 3: Backward through first linear layer
            // Gradient w.r.t. input
            dim3 gradInputGridSize = new dim3(
                (uint)((_modelDim + blockSize.x - 1) / blockSize.x),
                (uint)((totalRows + blockSize.y - 1) / blockSize.y));

            linearBackwardInputKernel.GridDimensions = gradInputGridSize;
            linearBackwardInputKernel.BlockDimensions = blockSize;
            linearBackwardInputKernel.Run(
                gradPreGeluDevice.DevicePointer,
                _w1Device.DevicePointer,
                gradInputDevice.DevicePointer,
                totalRows,
                _modelDim,
                _ffDim);

            // Gradient w.r.t. w1 and b1
            dim3 gradW1GridSize = new dim3(
                (uint)((_ffDim + blockSize.x - 1) / blockSize.x),
                (uint)((_modelDim + blockSize.y - 1) / blockSize.y));

            linearBackwardKernel.GridDimensions = gradW1GridSize;
            linearBackwardKernel.BlockDimensions = blockSize;
            linearBackwardKernel.Run(
                gradPreGeluDevice.DevicePointer,
                inputDevice.DevicePointer,
                _w1Device.DevicePointer,
                gradInputDevice.DevicePointer,  // Not used
                gradW1Device.DevicePointer,
                gradB1Device.DevicePointer,
                totalRows,
                _modelDim,
                _ffDim);

            // Copy gradients back to host
            float[] gradInputData = new float[inputSize];
            float[] gradW1Flat = new float[_modelDim * _ffDim];
            float[] gradW2Flat = new float[_ffDim * _modelDim];

            gradInputDevice.CopyToHost(gradInputData);
            gradW1Device.CopyToHost(gradW1Flat);
            gradW2Device.CopyToHost(gradW2Flat);
            gradB1Device.CopyToHost(_b1Grad);
            gradB2Device.CopyToHost(_b2Grad);

            // Unflatten gradients
            UnflattenTo(_w1Grad, gradW1Flat);
            UnflattenTo(_w2Grad, gradW2Flat);

            // Update device weights
            CopyWeightsToDevice();

            return new Tensor(gradInputData, gradOutput.Shape);
        }

        private void ApplyDropoutDevice(CudaDeviceVariable<float> data, int rows, int cols)
        {
            // Simple CPU-based dropout mask generation (GPU random is more complex)
            int size = rows * cols;
            float scale = 1.0f / (1.0f - _dropout);
            Random rng = new Random();

            float[] dataHost = new float[size];
            data.CopyToHost(dataHost);

            for (int i = 0; i < size; i++)
            {
                if (rng.NextDouble() < _dropout)
                {
                    dataHost[i] = 0f;
                }
                else
                {
                    dataHost[i] *= scale;
                }
            }

            data.CopyToDevice(dataHost);
        }

        private void UnflattenTo(float[,] target, float[] flat)
        {
            int rows = target.GetLength(0);
            int cols = target.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    target[i, j] = flat[i * cols + j];
                }
            }
        }

        private void ClearGradients()
        {
            Array.Clear(_w1Grad, 0, _w1Grad.Length);
            Array.Clear(_b1Grad, 0, _b1Grad.Length);
            Array.Clear(_w2Grad, 0, _w2Grad.Length);
            Array.Clear(_b2Grad, 0, _b2Grad.Length);
        }

        public override float[,] Call(float[,] inputs)
        {
            int seqLen = inputs.GetLength(0);
            int dim = inputs.GetLength(1);

            Tensor input3D = new Tensor(new[] { 1, seqLen, dim });
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < dim; d++)
                {
                    input3D[0, s, d] = inputs[s, d];
                }
            }

            Tensor output3D = Forward(input3D);

            float[,] output = new float[seqLen, _modelDim];
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < _modelDim; d++)
                {
                    output[s, d] = output3D[0, s, d];
                }
            }

            return output;
        }

        public override float[,] Backward(float[,] gradient)
        {
            int seqLen = gradient.GetLength(0);
            int dim = gradient.GetLength(1);

            Tensor grad3D = new Tensor(new[] { 1, seqLen, dim });
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < dim; d++)
                {
                    grad3D[0, s, d] = gradient[s, d];
                }
            }

            Tensor dInput3D = Backward(grad3D);

            float[,] dInput = new float[seqLen, _modelDim];
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < _modelDim; d++)
                {
                    dInput[s, d] = dInput3D[0, s, d];
                }
            }

            return dInput;
        }

        public override float[,,,] Call(float[,,,] inputs)
        {
            throw new NotImplementedException("FeedForwardCuda does not support 4D tensors");
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException("FeedForwardCuda does not support 4D tensors");
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return inputShape;
        }

        public override void Dispose()
        {
            _w1Device?.Dispose();
            _b1Device?.Dispose();
            _w2Device?.Dispose();
            _b2Device?.Dispose();
            _hiddenDevice?.Dispose();
            _preGeluDevice?.Dispose();

            if (_contextOwned)
            {
                _context?.Dispose();
            }

            GC.SuppressFinalize(this);
        }
    }
}
