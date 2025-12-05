using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using ManagedCuda;
using NeuralNetwork.Tensors;
using NeuralNetwork.Layers;
using NeuralNetwork.Layers.Cuda;
using NeuralNetwork.Initializers;

namespace NeuralNetwork.Models
{
    /// <summary>
    /// CUDA-accelerated Transformer Language Model (decoder-only, like GPT).
    ///
    /// Architecture:
    /// 1. Token Embedding (CUDA)
    /// 2. Positional Encoding
    /// 3. N x Transformer Blocks (CUDA)
    /// 4. Final Layer Normalization (CUDA)
    /// 5. Output Projection (CUDA)
    /// </summary>
    public class TransformerLMCuda : IDisposable
    {
        private readonly TransformerConfig _config;

        private CudaContext _context;
        private string _embeddingKernelPath;
        private string _denseKernelPath;

        // Model components
        private CudaDeviceVariable<float> _tokenEmbeddingsDevice;
        private float[,] _tokenEmbeddings;      // [vocabSize, embeddingDim] - host copy
        private PositionalEncoding _posEncoding;
        private Dropout _embeddingDropout;
        private TransformerBlockCuda[] _blocks;
        private LayerNormCuda _finalNorm;
        private CudaDeviceVariable<float> _outputProjectionDevice;
        private float[,] _outputProjection;     // [embeddingDim, vocabSize] - host copy

        // Gradient accumulators
        private float[,] _tokenEmbeddingsGrad;
        private float[,] _outputProjectionGrad;

        // Cached values for backward pass
        private int[] _lastTokens;
        private Tensor _lastEmbedded;
        private Tensor _lastAfterPosEnc;
        private Tensor[] _blockOutputs;
        private Tensor _lastNormOutput;

        private bool _training;

        public TransformerConfig Config => _config;

        public bool Training
        {
            get => _training;
            set
            {
                _training = value;
                _posEncoding.Training = value;
                if (_embeddingDropout != null) _embeddingDropout.Training = value;
                foreach (var block in _blocks)
                    block.Training = value;
            }
        }

        // Expose parameters for optimizer
        public float[,] TokenEmbeddings => _tokenEmbeddings;
        public float[,] TokenEmbeddingsGrad => _tokenEmbeddingsGrad;
        public float[,] OutputProjection => _outputProjection;
        public float[,] OutputProjectionGrad => _outputProjectionGrad;
        public TransformerBlockCuda[] Blocks => _blocks;
        public LayerNormCuda FinalNorm => _finalNorm;

        /// <summary>
        /// Create a CUDA-accelerated Transformer Language Model.
        /// </summary>
        public TransformerLMCuda(TransformerConfig config)
        {
            _config = config;
            config.Validate();
            _training = true;

            // Initialize CUDA context
            _context = new CudaContext();
            _embeddingKernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "EmbeddingKernel.ptx");
            _denseKernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "DenseKernel.ptx");

            Initialize();
        }

        private void Initialize()
        {
            // Token embeddings
            _tokenEmbeddings = InitializeEmbeddings(_config.VocabSize, _config.EmbeddingDim);
            _tokenEmbeddingsGrad = new float[_config.VocabSize, _config.EmbeddingDim];

            // Allocate on device
            _tokenEmbeddingsDevice = new CudaDeviceVariable<float>(_config.VocabSize * _config.EmbeddingDim);
            CopyEmbeddingsToDevice();

            // Positional encoding (CPU for now, could be moved to GPU)
            _posEncoding = new PositionalEncoding(_config.MaxSeqLen, _config.EmbeddingDim, _config.EmbeddingDropout);

            // Embedding dropout
            if (_config.EmbeddingDropout > 0f)
            {
                _embeddingDropout = new Dropout(_config.EmbeddingDropout);
            }

            // CUDA Transformer blocks
            _blocks = new TransformerBlockCuda[_config.NumLayers];
            for (int i = 0; i < _config.NumLayers; i++)
            {
                _blocks[i] = new TransformerBlockCuda(
                    _config.EmbeddingDim,
                    _config.NumHeads,
                    _config.FFNDim,
                    _config.DropoutRate,
                    _config.UseCausalMask,
                    _context
                );
                _blocks[i].Build(new[] { 1, _config.MaxSeqLen, _config.EmbeddingDim });
            }

            // CUDA Final layer normalization
            _finalNorm = new LayerNormCuda(_config.EmbeddingDim, context: _context);

            // Output projection
            if (_config.TieEmbeddings)
            {
                _outputProjection = null;
                _outputProjectionGrad = null;
                _outputProjectionDevice = null;
            }
            else
            {
                _outputProjection = Initializers.Initializers.GlorotUniform(_config.EmbeddingDim, _config.VocabSize);
                _outputProjectionGrad = new float[_config.EmbeddingDim, _config.VocabSize];
                _outputProjectionDevice = new CudaDeviceVariable<float>(_config.EmbeddingDim * _config.VocabSize);
                CopyOutputProjectionToDevice();
            }

            _blockOutputs = new Tensor[_config.NumLayers];
        }

        private void CopyEmbeddingsToDevice()
        {
            float[] flat = Flatten(_tokenEmbeddings);
            _tokenEmbeddingsDevice.CopyToDevice(flat);
        }

        private void CopyOutputProjectionToDevice()
        {
            if (_outputProjection != null && _outputProjectionDevice != null)
            {
                float[] flat = Flatten(_outputProjection);
                _outputProjectionDevice.CopyToDevice(flat);
            }
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

        private float[,] InitializeEmbeddings(int vocabSize, int dim)
        {
            float[,] embeddings = new float[vocabSize, dim];
            Random rng = new Random(42);
            float scale = (float)Math.Sqrt(1.0 / dim);

            for (int i = 0; i < vocabSize; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    double u1 = 1.0 - rng.NextDouble();
                    double u2 = 1.0 - rng.NextDouble();
                    double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                    embeddings[i, j] = (float)(normal * scale);
                }
            }

            return embeddings;
        }

        /// <summary>
        /// Forward pass using CUDA acceleration.
        /// </summary>
        /// <param name="tokens">Token indices [batch, seqLen].</param>
        /// <returns>Logits [batch, seqLen, vocabSize].</returns>
        public Tensor Forward(int[,] tokens)
        {
            int batch = tokens.GetLength(0);
            int seqLen = tokens.GetLength(1);

            if (seqLen > _config.MaxSeqLen)
                throw new ArgumentException($"Sequence length {seqLen} exceeds maximum {_config.MaxSeqLen}");

            // Store for backward
            _lastTokens = Flatten2DArray(tokens);

            // Token embedding lookup (using existing CUDA embedding kernel)
            Tensor embedded = EmbeddingLookupCuda(tokens);
            _lastEmbedded = embedded;

            // Add positional encoding (CPU for now)
            Tensor afterPosEnc = _posEncoding.Forward(embedded);
            _lastAfterPosEnc = afterPosEnc;

            // Apply embedding dropout
            if (_embeddingDropout != null && _training)
            {
                afterPosEnc = ApplyDropout3D(afterPosEnc, _embeddingDropout);
            }

            // Pass through CUDA Transformer blocks
            Tensor hidden = afterPosEnc;
            for (int i = 0; i < _config.NumLayers; i++)
            {
                hidden = _blocks[i].Forward(hidden);
                _blockOutputs[i] = hidden;
            }

            // Final layer normalization (CUDA)
            Tensor normOutput = _finalNorm.Forward(hidden);
            _lastNormOutput = normOutput;

            // Output projection to vocabulary logits
            Tensor logits = ComputeOutputProjectionCuda(normOutput);

            return logits;
        }

        private Tensor EmbeddingLookupCuda(int[,] tokens)
        {
            int batch = tokens.GetLength(0);
            int seqLen = tokens.GetLength(1);
            int embDim = _config.EmbeddingDim;
            int totalTokens = batch * seqLen;

            // Convert tokens to flat array
            int[] tokenFlat = new int[totalTokens];
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    tokenFlat[b * seqLen + s] = tokens[b, s];
                }
            }

            using var tokensDevice = new CudaDeviceVariable<int>(totalTokens);
            using var outputDevice = new CudaDeviceVariable<float>(totalTokens * embDim);

            tokensDevice.CopyToDevice(tokenFlat);

            // Load and run embedding lookup kernel (using integer token version)
            var kernel = _context.LoadKernelPTX(_embeddingKernelPath, "EmbeddingLookupInt");

            int blockSize = 256;
            int gridSize = (totalTokens + blockSize - 1) / blockSize;

            kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((uint)gridSize, 1, 1);
            kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3((uint)blockSize, 1, 1);
            kernel.Run(
                tokensDevice.DevicePointer,
                _tokenEmbeddingsDevice.DevicePointer,
                outputDevice.DevicePointer,
                (int)totalTokens,
                (int)embDim,
                (int)_config.VocabSize);

            float[] outputData = new float[totalTokens * embDim];
            outputDevice.CopyToHost(outputData);

            return new Tensor(outputData, new[] { batch, seqLen, embDim });
        }

        private Tensor ComputeOutputProjectionCuda(Tensor hidden)
        {
            int batch = hidden.Shape[0];
            int seqLen = hidden.Shape[1];
            int vocabSize = _config.VocabSize;
            int embDim = _config.EmbeddingDim;
            int totalRows = batch * seqLen;

            Tensor logits = new Tensor(new[] { batch, seqLen, vocabSize });

            if (_config.TieEmbeddings)
            {
                // Use embeddings transposed: hidden @ embeddings^T
                // This needs a custom kernel for efficiency, using CPU for now
                Parallel.For(0, totalRows, row =>
                {
                    int hiddenOffset = row * embDim;
                    int logitsOffset = row * vocabSize;

                    for (int v = 0; v < vocabSize; v++)
                    {
                        float sum = 0f;
                        for (int d = 0; d < embDim; d++)
                        {
                            sum += hidden.Data[hiddenOffset + d] * _tokenEmbeddings[v, d];
                        }
                        logits.Data[logitsOffset + v] = sum;
                    }
                });
            }
            else
            {
                // Use separate output projection with CUDA
                using var hiddenDevice = new CudaDeviceVariable<float>(totalRows * embDim);
                using var logitsDevice = new CudaDeviceVariable<float>(totalRows * vocabSize);

                hiddenDevice.CopyToDevice(hidden.Data);

                var matmulKernel = _context.LoadKernelPTX(_denseKernelPath, "MatMul");
                var blockSize = new ManagedCuda.VectorTypes.dim3(16, 16);
                var gridSize = new ManagedCuda.VectorTypes.dim3(
                    (uint)((vocabSize + blockSize.x - 1) / blockSize.x),
                    (uint)((totalRows + blockSize.y - 1) / blockSize.y));

                matmulKernel.GridDimensions = gridSize;
                matmulKernel.BlockDimensions = blockSize;
                matmulKernel.Run(
                    hiddenDevice.DevicePointer,
                    _outputProjectionDevice.DevicePointer,
                    logitsDevice.DevicePointer,
                    totalRows, embDim, vocabSize);

                logitsDevice.CopyToHost(logits.Data);
            }

            return logits;
        }

        /// <summary>
        /// Compute loss (cross-entropy) for language modeling.
        /// </summary>
        public float ComputeLoss(Tensor logits, int[,] targets)
        {
            int batch = logits.Shape[0];
            int seqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];

            float totalLoss = 0f;
            int count = 0;

            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    float maxLogit = float.NegativeInfinity;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        if (logits[b, s, v] > maxLogit)
                            maxLogit = logits[b, s, v];
                    }

                    float sumExp = 0f;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        sumExp += (float)Math.Exp(logits[b, s, v] - maxLogit);
                    }

                    int target = targets[b, s];
                    float logProb = logits[b, s, target] - maxLogit - (float)Math.Log(sumExp);
                    totalLoss -= logProb;
                    count++;
                }
            }

            return totalLoss / count;
        }

        /// <summary>
        /// Backward pass.
        /// </summary>
        public void Backward(Tensor logits, int[,] targets)
        {
            int batch = logits.Shape[0];
            int seqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];

            // Compute gradient of cross-entropy loss w.r.t. logits
            Tensor dLogits = ComputeCrossEntropyGradient(logits, targets);

            // Clear embedding gradients
            Array.Clear(_tokenEmbeddingsGrad, 0, _tokenEmbeddingsGrad.Length);
            if (_outputProjectionGrad != null)
                Array.Clear(_outputProjectionGrad, 0, _outputProjectionGrad.Length);

            // Gradient through output projection
            Tensor dNormOutput = OutputProjectionBackward(dLogits, _lastNormOutput);

            // Gradient through final layer norm
            Tensor dHidden = _finalNorm.Backward(dNormOutput);

            // Gradient through Transformer blocks (reverse order)
            for (int i = _config.NumLayers - 1; i >= 0; i--)
            {
                Tensor blockInput = i == 0 ? _lastAfterPosEnc : _blockOutputs[i - 1];

                // Re-run forward to set up cached values
                if (i > 0)
                {
                    Tensor prevHidden = _lastAfterPosEnc;
                    for (int j = 0; j < i; j++)
                    {
                        prevHidden = _blocks[j].Forward(prevHidden);
                    }
                }
                _blocks[i].Forward(blockInput);
                dHidden = _blocks[i].Backward(dHidden);
            }

            // Gradient through embedding dropout
            if (_embeddingDropout != null && _training)
            {
                dHidden = ApplyDropoutBackward3D(dHidden, _embeddingDropout);
            }

            // Gradient through positional encoding
            Tensor dEmbedded = _posEncoding.Backward(dHidden);

            // Gradient through token embeddings
            EmbeddingBackward(dEmbedded);

            // Sync updated weights to device
            CopyEmbeddingsToDevice();
            CopyOutputProjectionToDevice();
        }

        private Tensor ComputeCrossEntropyGradient(Tensor logits, int[,] targets)
        {
            int batch = logits.Shape[0];
            int seqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];
            int totalRows = batch * seqLen;
            float scale = 1.0f / totalRows;

            Tensor gradient = new Tensor(logits.Shape);

            Parallel.For(0, totalRows, row =>
            {
                int b = row / seqLen;
                int s = row % seqLen;
                int logitsOffset = row * vocabSize;

                float maxLogit = float.NegativeInfinity;
                for (int v = 0; v < vocabSize; v++)
                {
                    float val = logits.Data[logitsOffset + v];
                    if (val > maxLogit)
                        maxLogit = val;
                }

                float sumExp = 0f;
                for (int v = 0; v < vocabSize; v++)
                {
                    sumExp += MathF.Exp(logits.Data[logitsOffset + v] - maxLogit);
                }

                int target = targets[b, s];
                for (int v = 0; v < vocabSize; v++)
                {
                    float softmax = MathF.Exp(logits.Data[logitsOffset + v] - maxLogit) / sumExp;
                    gradient.Data[logitsOffset + v] = (softmax - (v == target ? 1f : 0f)) * scale;
                }
            });

            return gradient;
        }

        private Tensor OutputProjectionBackward(Tensor dLogits, Tensor hidden)
        {
            int batch = dLogits.Shape[0];
            int seqLen = dLogits.Shape[1];
            int vocabSize = _config.VocabSize;
            int embDim = _config.EmbeddingDim;
            int totalRows = batch * seqLen;

            Tensor dHidden = new Tensor(hidden.Shape);

            if (_config.TieEmbeddings)
            {
                var localEmbGrads = new System.Threading.ThreadLocal<float[,]>(
                    () => new float[vocabSize, embDim], trackAllValues: true);

                Parallel.For(0, totalRows, row =>
                {
                    int logitsOffset = row * vocabSize;
                    int hiddenOffset = row * embDim;
                    var localGrad = localEmbGrads.Value;

                    for (int d = 0; d < embDim; d++)
                    {
                        float sum = 0f;
                        float hiddenVal = hidden.Data[hiddenOffset + d];
                        for (int v = 0; v < vocabSize; v++)
                        {
                            float dLogitVal = dLogits.Data[logitsOffset + v];
                            sum += dLogitVal * _tokenEmbeddings[v, d];
                            localGrad[v, d] += dLogitVal * hiddenVal;
                        }
                        dHidden.Data[hiddenOffset + d] = sum;
                    }
                });

                foreach (var localGrad in localEmbGrads.Values)
                {
                    for (int v = 0; v < vocabSize; v++)
                    {
                        for (int d = 0; d < embDim; d++)
                        {
                            _tokenEmbeddingsGrad[v, d] += localGrad[v, d];
                        }
                    }
                }
                localEmbGrads.Dispose();
            }
            else
            {
                var localProjGrads = new System.Threading.ThreadLocal<float[,]>(
                    () => new float[embDim, vocabSize], trackAllValues: true);

                Parallel.For(0, totalRows, row =>
                {
                    int logitsOffset = row * vocabSize;
                    int hiddenOffset = row * embDim;
                    var localGrad = localProjGrads.Value;

                    for (int d = 0; d < embDim; d++)
                    {
                        float sum = 0f;
                        float hiddenVal = hidden.Data[hiddenOffset + d];
                        for (int v = 0; v < vocabSize; v++)
                        {
                            float dLogitVal = dLogits.Data[logitsOffset + v];
                            sum += dLogitVal * _outputProjection[d, v];
                            localGrad[d, v] += dLogitVal * hiddenVal;
                        }
                        dHidden.Data[hiddenOffset + d] = sum;
                    }
                });

                foreach (var localGrad in localProjGrads.Values)
                {
                    for (int d = 0; d < embDim; d++)
                    {
                        for (int v = 0; v < vocabSize; v++)
                        {
                            _outputProjectionGrad[d, v] += localGrad[d, v];
                        }
                    }
                }
                localProjGrads.Dispose();
            }

            return dHidden;
        }

        private void EmbeddingBackward(Tensor dEmbedded)
        {
            int batch = dEmbedded.Shape[0];
            int seqLen = dEmbedded.Shape[1];

            for (int i = 0; i < _lastTokens.Length; i++)
            {
                int b = i / seqLen;
                int s = i % seqLen;
                int tokenId = _lastTokens[i];

                for (int d = 0; d < _config.EmbeddingDim; d++)
                {
                    _tokenEmbeddingsGrad[tokenId, d] += dEmbedded[b, s, d];
                }
            }
        }

        private Tensor ApplyDropout3D(Tensor input, Dropout dropout)
        {
            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int dim = input.Shape[2];

            float[,] input2D = new float[batch * seqLen, dim];
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        input2D[b * seqLen + s, d] = input[b, s, d];
                    }
                }
            }

            if (!dropout.Built)
                dropout.Build(new[] { batch * seqLen, dim });

            float[,] dropped = dropout.Call(input2D);

            Tensor output = new Tensor(input.Shape);
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        output[b, s, d] = dropped[b * seqLen + s, d];
                    }
                }
            }

            return output;
        }

        private Tensor ApplyDropoutBackward3D(Tensor gradOutput, Dropout dropout)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];
            int dim = gradOutput.Shape[2];

            float[,] grad2D = new float[batch * seqLen, dim];
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        grad2D[b * seqLen + s, d] = gradOutput[b, s, d];
                    }
                }
            }

            float[,] dInput2D = dropout.Backward(grad2D);

            Tensor dInput = new Tensor(gradOutput.Shape);
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        dInput[b, s, d] = dInput2D[b * seqLen + s, d];
                    }
                }
            }

            return dInput;
        }

        private int[] Flatten2DArray(int[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            int[] flat = new int[rows * cols];

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
        /// Get all trainable parameters for optimizer.
        /// </summary>
        public List<(string name, float[,] weights, float[,] gradients)> GetParameters()
        {
            var parameters = new List<(string, float[,], float[,])>();

            parameters.Add(("token_embeddings", _tokenEmbeddings, _tokenEmbeddingsGrad));

            for (int i = 0; i < _blocks.Length; i++)
            {
                var block = _blocks[i];
                var (weights, biases, weightGrads, biasGrads) = block.GetParameters();

                for (int w = 0; w < weights.Count; w++)
                {
                    parameters.Add(($"block_{i}.weight_{w}", weights[w], weightGrads[w]));
                }
            }

            if (!_config.TieEmbeddings && _outputProjection != null)
            {
                parameters.Add(("output_projection", _outputProjection, _outputProjectionGrad));
            }

            return parameters;
        }

        /// <summary>
        /// Count total trainable parameters.
        /// </summary>
        public long CountParameters()
        {
            long total = 0;

            total += _tokenEmbeddings.Length;

            foreach (var block in _blocks)
            {
                var (weights, biases, _, _) = block.GetParameters();
                foreach (var w in weights)
                    total += w.Length;
                foreach (var b in biases)
                    total += b.Length;
            }

            total += 2 * _config.EmbeddingDim; // Final layer norm

            if (!_config.TieEmbeddings && _outputProjection != null)
            {
                total += _outputProjection.Length;
            }

            return total;
        }

        /// <summary>
        /// Synchronize all parameter updates to CUDA device memory.
        /// Call this after optimizer updates weights.
        /// </summary>
        public void SyncToDevice()
        {
            CopyEmbeddingsToDevice();
            CopyOutputProjectionToDevice();
            _finalNorm.SyncParametersToDevice();
        }

        public void Dispose()
        {
            _tokenEmbeddingsDevice?.Dispose();
            _outputProjectionDevice?.Dispose();

            foreach (var block in _blocks)
            {
                block?.Dispose();
            }

            _finalNorm?.Dispose();
            _context?.Dispose();
        }
    }
}
