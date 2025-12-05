using System;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Layers.Cuda
{
    /// <summary>
    /// CUDA-accelerated Scaled Dot-Product Attention.
    /// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    /// </summary>
    public class ScaledDotProductAttentionCuda : IDisposable
    {
        private readonly float _scale;
        private readonly bool _useCausalMask;
        private readonly float _dropout;
        private readonly Random _rng;

        private CudaContext _context;
        private bool _contextOwned;

        // Cached device variables for intermediate results
        private CudaDeviceVariable<float> _scoresDevice;
        private CudaDeviceVariable<float> _attnWeightsDevice;

        // Cached values for backward pass
        private Tensor _lastQ;
        private Tensor _lastK;
        private Tensor _lastV;
        private Tensor _lastAttnWeights;
        private Tensor _lastDropoutMask;
        private bool _training;

        // Kernel path
        private string _kernelPath;

        // Track allocated sizes for reuse
        private int _allocatedScoresSize;
        private int _allocatedAttnWeightsSize;

        /// <summary>
        /// Create a CUDA-accelerated Scaled Dot-Product Attention module.
        /// </summary>
        /// <param name="headDim">Dimension of each attention head (d_k).</param>
        /// <param name="useCausalMask">Whether to apply causal masking.</param>
        /// <param name="dropout">Dropout rate for attention weights.</param>
        /// <param name="context">Optional shared CUDA context.</param>
        public ScaledDotProductAttentionCuda(int headDim, bool useCausalMask = true, float dropout = 0f, CudaContext context = null)
        {
            _scale = 1.0f / (float)Math.Sqrt(headDim);
            _useCausalMask = useCausalMask;
            _dropout = dropout;
            _rng = new Random();
            _training = true;

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

            _kernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "AttentionKernel.ptx");
        }

        /// <summary>
        /// Whether the module is in training mode.
        /// </summary>
        public bool Training
        {
            get => _training;
            set => _training = value;
        }

        /// <summary>
        /// Forward pass using CUDA.
        /// </summary>
        /// <param name="query">Query tensor [batch, seqLen, headDim].</param>
        /// <param name="key">Key tensor [batch, seqLen, headDim].</param>
        /// <param name="value">Value tensor [batch, seqLen, headDim].</param>
        /// <param name="mask">Optional additional mask [batch, seqLen, seqLen].</param>
        /// <returns>Attention output [batch, seqLen, headDim].</returns>
        public Tensor Forward(Tensor query, Tensor key, Tensor value, Tensor mask = null)
        {
            if (query.Rank != 3 || key.Rank != 3 || value.Rank != 3)
                throw new ArgumentException("Q, K, V must be rank-3 tensors [batch, seqLen, dim]");

            int batch = query.Shape[0];
            int seqLen = query.Shape[1];
            int headDim = query.Shape[2];

            // Cache for backward
            _lastQ = query;
            _lastK = key;
            _lastV = value;

            // Allocate device memory
            int qkvSize = batch * seqLen * headDim;
            int scoresSize = batch * seqLen * seqLen;

            using var qDevice = new CudaDeviceVariable<float>(qkvSize);
            using var kDevice = new CudaDeviceVariable<float>(qkvSize);
            using var vDevice = new CudaDeviceVariable<float>(qkvSize);
            using var scoresDevice = new CudaDeviceVariable<float>(scoresSize);
            using var attnWeightsDevice = new CudaDeviceVariable<float>(scoresSize);
            using var outputDevice = new CudaDeviceVariable<float>(qkvSize);

            // Copy inputs to device
            qDevice.CopyToDevice(query.Data);
            kDevice.CopyToDevice(key.Data);
            vDevice.CopyToDevice(value.Data);

            // Step 1: Compute QK^T using batched matmul with K transposed
            // Q: [batch, seqLen, headDim], K: [batch, seqLen, headDim]
            // Result: [batch, seqLen, seqLen]
            var matmulKernel = _context.LoadKernelPTX(_kernelPath, "BatchedMatMulTransposeB");
            dim3 blockSize = new dim3(16, 16, 1);
            dim3 gridSize = new dim3(
                (uint)((seqLen + blockSize.x - 1) / blockSize.x),
                (uint)((seqLen + blockSize.y - 1) / blockSize.y),
                (uint)batch);

            matmulKernel.GridDimensions = gridSize;
            matmulKernel.BlockDimensions = blockSize;
            matmulKernel.Run(
                qDevice.DevicePointer,
                kDevice.DevicePointer,
                scoresDevice.DevicePointer,
                batch, seqLen, headDim, seqLen);

            // Step 2: Scale scores
            var scaleKernel = _context.LoadKernelPTX(_kernelPath, "Scale");
            int scaleBlockSize = 256;
            int scaleGridSize = (scoresSize + scaleBlockSize - 1) / scaleBlockSize;

            scaleKernel.GridDimensions = new dim3((uint)scaleGridSize, 1, 1);
            scaleKernel.BlockDimensions = new dim3((uint)scaleBlockSize, 1, 1);
            scaleKernel.Run(scoresDevice.DevicePointer, _scale, scoresSize);

            // Step 3: Apply causal mask if needed
            if (_useCausalMask)
            {
                var maskKernel = _context.LoadKernelPTX(_kernelPath, "ApplyCausalMask");
                maskKernel.GridDimensions = new dim3(
                    (uint)((seqLen + 15) / 16),
                    (uint)((seqLen + 15) / 16),
                    (uint)batch);
                maskKernel.BlockDimensions = new dim3(16, 16, 1);
                maskKernel.Run(scoresDevice.DevicePointer, batch, seqLen);
            }

            // Step 4: Softmax along last dimension
            var softmaxKernel = _context.LoadKernelPTX(_kernelPath, "Softmax3D");
            // Each block processes one row (one query position for one batch)
            int softmaxBlockSize = Math.Min(256, seqLen);
            // Ensure power of 2 for reduction
            softmaxBlockSize = (int)Math.Pow(2, Math.Ceiling(Math.Log(softmaxBlockSize) / Math.Log(2)));
            softmaxBlockSize = Math.Max(32, Math.Min(256, softmaxBlockSize));

            softmaxKernel.GridDimensions = new dim3((uint)(batch * seqLen), 1, 1);
            softmaxKernel.BlockDimensions = new dim3((uint)softmaxBlockSize, 1, 1);
            softmaxKernel.DynamicSharedMemory = (uint)(softmaxBlockSize * sizeof(float));
            softmaxKernel.Run(
                scoresDevice.DevicePointer,
                attnWeightsDevice.DevicePointer,
                batch, seqLen, seqLen);

            // Cache attention weights for backward
            float[] attnWeightsData = new float[scoresSize];
            attnWeightsDevice.CopyToHost(attnWeightsData);
            _lastAttnWeights = new Tensor(attnWeightsData, new[] { batch, seqLen, seqLen });

            // Step 5: Apply dropout if training
            if (_training && _dropout > 0f)
            {
                ApplyDropoutDevice(attnWeightsDevice, batch, seqLen, seqLen);
            }

            // Step 6: Compute attnWeights @ V
            // attnWeights: [batch, seqLen, seqLen], V: [batch, seqLen, headDim]
            // Result: [batch, seqLen, headDim]
            var outputMatmulKernel = _context.LoadKernelPTX(_kernelPath, "BatchedMatMul");
            dim3 outBlockSize = new dim3(16, 16, 1);
            dim3 outGridSize = new dim3(
                (uint)((headDim + outBlockSize.x - 1) / outBlockSize.x),
                (uint)((seqLen + outBlockSize.y - 1) / outBlockSize.y),
                (uint)batch);

            outputMatmulKernel.GridDimensions = outGridSize;
            outputMatmulKernel.BlockDimensions = outBlockSize;
            outputMatmulKernel.Run(
                attnWeightsDevice.DevicePointer,
                vDevice.DevicePointer,
                outputDevice.DevicePointer,
                batch, seqLen, seqLen, headDim);

            // Copy result back to host
            float[] outputData = new float[qkvSize];
            outputDevice.CopyToHost(outputData);

            return new Tensor(outputData, new[] { batch, seqLen, headDim });
        }

        /// <summary>
        /// Backward pass using CUDA.
        /// </summary>
        /// <param name="gradOutput">Gradient w.r.t. output [batch, seqLen, headDim].</param>
        /// <returns>Gradients w.r.t. Q, K, V.</returns>
        public (Tensor dQ, Tensor dK, Tensor dV) Backward(Tensor gradOutput)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];
            int headDim = gradOutput.Shape[2];

            int qkvSize = batch * seqLen * headDim;
            int scoresSize = batch * seqLen * seqLen;

            using var gradOutputDevice = new CudaDeviceVariable<float>(qkvSize);
            using var attnWeightsDevice = new CudaDeviceVariable<float>(scoresSize);
            using var vDevice = new CudaDeviceVariable<float>(qkvSize);
            using var qDevice = new CudaDeviceVariable<float>(qkvSize);
            using var kDevice = new CudaDeviceVariable<float>(qkvSize);
            using var dVDevice = new CudaDeviceVariable<float>(qkvSize);
            using var dAttnWeightsDevice = new CudaDeviceVariable<float>(scoresSize);
            using var dScoresDevice = new CudaDeviceVariable<float>(scoresSize);
            using var dQDevice = new CudaDeviceVariable<float>(qkvSize);
            using var dKDevice = new CudaDeviceVariable<float>(qkvSize);

            // Copy to device
            gradOutputDevice.CopyToDevice(gradOutput.Data);
            attnWeightsDevice.CopyToDevice(_lastAttnWeights.Data);
            vDevice.CopyToDevice(_lastV.Data);
            qDevice.CopyToDevice(_lastQ.Data);
            kDevice.CopyToDevice(_lastK.Data);

            dim3 blockSize = new dim3(16, 16, 1);

            // Step 1: dV = attnWeights^T @ gradOutput
            // Need to transpose attnWeights: [batch, seqLen, seqLen] -> use BatchedMatMulTransposeA
            var dVKernel = _context.LoadKernelPTX(_kernelPath, "BatchedMatMulTransposeA");
            dim3 dVGridSize = new dim3(
                (uint)((headDim + blockSize.x - 1) / blockSize.x),
                (uint)((seqLen + blockSize.y - 1) / blockSize.y),
                (uint)batch);

            dVKernel.GridDimensions = dVGridSize;
            dVKernel.BlockDimensions = blockSize;
            dVKernel.Run(
                attnWeightsDevice.DevicePointer,
                gradOutputDevice.DevicePointer,
                dVDevice.DevicePointer,
                batch, seqLen, seqLen, headDim);

            // Step 2: dAttnWeights = gradOutput @ V^T
            var dAttnKernel = _context.LoadKernelPTX(_kernelPath, "BatchedMatMulTransposeB");
            dim3 dAttnGridSize = new dim3(
                (uint)((seqLen + blockSize.x - 1) / blockSize.x),
                (uint)((seqLen + blockSize.y - 1) / blockSize.y),
                (uint)batch);

            dAttnKernel.GridDimensions = dAttnGridSize;
            dAttnKernel.BlockDimensions = blockSize;
            dAttnKernel.Run(
                gradOutputDevice.DevicePointer,
                vDevice.DevicePointer,
                dAttnWeightsDevice.DevicePointer,
                batch, seqLen, headDim, seqLen);

            // Apply dropout gradient
            if (_training && _dropout > 0f && _lastDropoutMask != null)
            {
                ApplyDropoutMaskDevice(dAttnWeightsDevice, batch, seqLen, seqLen);
            }

            // Step 3: Softmax backward
            var softmaxBackwardKernel = _context.LoadKernelPTX(_kernelPath, "SoftmaxBackward3D");
            int softmaxBlockSize = Math.Min(256, seqLen);
            softmaxBlockSize = (int)Math.Pow(2, Math.Ceiling(Math.Log(softmaxBlockSize) / Math.Log(2)));
            softmaxBlockSize = Math.Max(32, Math.Min(256, softmaxBlockSize));

            softmaxBackwardKernel.GridDimensions = new dim3((uint)seqLen, (uint)batch, 1);
            softmaxBackwardKernel.BlockDimensions = new dim3((uint)softmaxBlockSize, 1, 1);
            softmaxBackwardKernel.DynamicSharedMemory = (uint)(softmaxBlockSize * sizeof(float));
            softmaxBackwardKernel.Run(
                dAttnWeightsDevice.DevicePointer,
                attnWeightsDevice.DevicePointer,
                dScoresDevice.DevicePointer,
                batch, seqLen, seqLen);

            // Step 4: Scale gradient
            var scaleKernel = _context.LoadKernelPTX(_kernelPath, "Scale");
            int scaleBlockSize2 = 256;
            int scaleGridSize = (scoresSize + scaleBlockSize2 - 1) / scaleBlockSize2;

            scaleKernel.GridDimensions = new dim3((uint)scaleGridSize, 1, 1);
            scaleKernel.BlockDimensions = new dim3((uint)scaleBlockSize2, 1, 1);
            scaleKernel.Run(dScoresDevice.DevicePointer, _scale, scoresSize);

            // Step 5: dQ = dScores @ K
            var dQKernel = _context.LoadKernelPTX(_kernelPath, "BatchedMatMul");
            dim3 dQGridSize = new dim3(
                (uint)((headDim + blockSize.x - 1) / blockSize.x),
                (uint)((seqLen + blockSize.y - 1) / blockSize.y),
                (uint)batch);

            dQKernel.GridDimensions = dQGridSize;
            dQKernel.BlockDimensions = blockSize;
            dQKernel.Run(
                dScoresDevice.DevicePointer,
                kDevice.DevicePointer,
                dQDevice.DevicePointer,
                batch, seqLen, seqLen, headDim);

            // Step 6: dK = dScores^T @ Q
            var dKKernel = _context.LoadKernelPTX(_kernelPath, "BatchedMatMulTransposeA");
            dim3 dKGridSize = new dim3(
                (uint)((headDim + blockSize.x - 1) / blockSize.x),
                (uint)((seqLen + blockSize.y - 1) / blockSize.y),
                (uint)batch);

            dKKernel.GridDimensions = dKGridSize;
            dKKernel.BlockDimensions = blockSize;
            dKKernel.Run(
                dScoresDevice.DevicePointer,
                qDevice.DevicePointer,
                dKDevice.DevicePointer,
                batch, seqLen, seqLen, headDim);

            // Copy results back to host
            float[] dQData = new float[qkvSize];
            float[] dKData = new float[qkvSize];
            float[] dVData = new float[qkvSize];

            dQDevice.CopyToHost(dQData);
            dKDevice.CopyToHost(dKData);
            dVDevice.CopyToHost(dVData);

            return (
                new Tensor(dQData, new[] { batch, seqLen, headDim }),
                new Tensor(dKData, new[] { batch, seqLen, headDim }),
                new Tensor(dVData, new[] { batch, seqLen, headDim })
            );
        }

        private void ApplyDropoutDevice(CudaDeviceVariable<float> data, int batch, int rows, int cols)
        {
            int size = batch * rows * cols;
            float scale = 1.0f / (1.0f - _dropout);

            // Generate dropout mask on CPU (GPU random is more complex)
            float[] mask = new float[size];
            for (int i = 0; i < size; i++)
            {
                mask[i] = _rng.NextDouble() >= _dropout ? scale : 0f;
            }

            _lastDropoutMask = new Tensor(mask, new[] { batch, rows, cols });

            // Apply mask on GPU
            using var maskDevice = new CudaDeviceVariable<float>(size);
            using var resultDevice = new CudaDeviceVariable<float>(size);

            maskDevice.CopyToDevice(mask);

            var multiplyKernel = _context.LoadKernelPTX(_kernelPath, "ElementwiseMultiply");
            int blockSize = 256;
            int gridSize = (size + blockSize - 1) / blockSize;

            multiplyKernel.GridDimensions = new dim3((uint)gridSize, 1, 1);
            multiplyKernel.BlockDimensions = new dim3((uint)blockSize, 1, 1);
            multiplyKernel.Run(data.DevicePointer, maskDevice.DevicePointer, data.DevicePointer, size);
        }

        private void ApplyDropoutMaskDevice(CudaDeviceVariable<float> data, int batch, int rows, int cols)
        {
            if (_lastDropoutMask == null) return;

            int size = batch * rows * cols;

            using var maskDevice = new CudaDeviceVariable<float>(size);
            maskDevice.CopyToDevice(_lastDropoutMask.Data);

            var multiplyKernel = _context.LoadKernelPTX(_kernelPath, "ElementwiseMultiply");
            int blockSize = 256;
            int gridSize = (size + blockSize - 1) / blockSize;

            multiplyKernel.GridDimensions = new dim3((uint)gridSize, 1, 1);
            multiplyKernel.BlockDimensions = new dim3((uint)blockSize, 1, 1);
            multiplyKernel.Run(data.DevicePointer, maskDevice.DevicePointer, data.DevicePointer, size);
        }

        public void Dispose()
        {
            _scoresDevice?.Dispose();
            _attnWeightsDevice?.Dispose();

            if (_contextOwned)
            {
                _context?.Dispose();
            }
        }
    }
}
