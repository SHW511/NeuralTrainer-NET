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
    /// CUDA-accelerated Multi-Head Attention.
    /// MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
    /// where head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)
    /// </summary>
    public class MultiHeadAttentionCuda : Layer
    {
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _modelDim;
        private readonly float _dropout;
        private readonly bool _useCausalMask;

        private CudaContext _context;
        private bool _contextOwned;
        private string _kernelPath;
        private string _denseKernelPath;

        // Projection weights on device
        private CudaDeviceVariable<float> _wQDevice;
        private CudaDeviceVariable<float> _wKDevice;
        private CudaDeviceVariable<float> _wVDevice;
        private CudaDeviceVariable<float> _wODevice;
        private CudaDeviceVariable<float> _bQDevice;
        private CudaDeviceVariable<float> _bKDevice;
        private CudaDeviceVariable<float> _bVDevice;
        private CudaDeviceVariable<float> _bODevice;

        // Host copies for optimizer access
        private float[,] _wQ;
        private float[,] _wK;
        private float[,] _wV;
        private float[,] _wO;
        private float[] _bQ;
        private float[] _bK;
        private float[] _bV;
        private float[] _bO;

        // Gradient accumulators
        private float[,] _wQGrad;
        private float[,] _wKGrad;
        private float[,] _wVGrad;
        private float[,] _wOGrad;
        private float[] _bQGrad;
        private float[] _bKGrad;
        private float[] _bVGrad;
        private float[] _bOGrad;

        // Cached values for backward pass
        private Tensor _lastInput;
        private Tensor _lastQ;
        private Tensor _lastK;
        private Tensor _lastV;
        private Tensor _lastAttnOutput;

        // Per-head attention modules
        private ScaledDotProductAttentionCuda[] _attentionHeads;

        private bool _training;

        public bool Training
        {
            get => _training;
            set
            {
                _training = value;
                if (_attentionHeads != null)
                {
                    foreach (var head in _attentionHeads)
                        head.Training = value;
                }
            }
        }

        // Expose weights for optimizer
        public float[,] WQ => _wQ;
        public float[,] WK => _wK;
        public float[,] WV => _wV;
        public float[,] WO => _wO;
        public float[,] WQGrad => _wQGrad;
        public float[,] WKGrad => _wKGrad;
        public float[,] WVGrad => _wVGrad;
        public float[,] WOGrad => _wOGrad;
        public float[] BQGrad => _bQGrad;
        public float[] BKGrad => _bKGrad;
        public float[] BVGrad => _bVGrad;
        public float[] BOGrad => _bOGrad;

        /// <summary>
        /// Create a CUDA-accelerated Multi-Head Attention layer.
        /// </summary>
        public MultiHeadAttentionCuda(int modelDim, int numHeads, float dropout = 0f,
            bool useCausalMask = true, CudaContext context = null)
        {
            if (modelDim % numHeads != 0)
                throw new ArgumentException($"Model dimension {modelDim} must be divisible by number of heads {numHeads}");

            _modelDim = modelDim;
            _numHeads = numHeads;
            _headDim = modelDim / numHeads;
            _dropout = dropout;
            _useCausalMask = useCausalMask;
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

            _kernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "AttentionKernel.ptx");
            _denseKernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "DenseKernel.ptx");
        }

        public override void Build(int[] inputShape)
        {
            if (Built) return;

            InputShape = inputShape;
            InputDim = inputShape[inputShape.Length - 1];

            if (InputDim != _modelDim)
                throw new ArgumentException($"Input dimension {InputDim} doesn't match model dimension {_modelDim}");

            // Initialize projection weights using Glorot uniform
            _wQ = Initializers.Initializers.GlorotUniform(_modelDim, _modelDim);
            _wK = Initializers.Initializers.GlorotUniform(_modelDim, _modelDim);
            _wV = Initializers.Initializers.GlorotUniform(_modelDim, _modelDim);
            _wO = Initializers.Initializers.GlorotUniform(_modelDim, _modelDim);

            // Initialize biases to zero
            _bQ = new float[_modelDim];
            _bK = new float[_modelDim];
            _bV = new float[_modelDim];
            _bO = new float[_modelDim];

            // Initialize gradient accumulators
            _wQGrad = new float[_modelDim, _modelDim];
            _wKGrad = new float[_modelDim, _modelDim];
            _wVGrad = new float[_modelDim, _modelDim];
            _wOGrad = new float[_modelDim, _modelDim];
            _bQGrad = new float[_modelDim];
            _bKGrad = new float[_modelDim];
            _bVGrad = new float[_modelDim];
            _bOGrad = new float[_modelDim];

            // Allocate device memory for weights
            int weightSize = _modelDim * _modelDim;
            _wQDevice = new CudaDeviceVariable<float>(weightSize);
            _wKDevice = new CudaDeviceVariable<float>(weightSize);
            _wVDevice = new CudaDeviceVariable<float>(weightSize);
            _wODevice = new CudaDeviceVariable<float>(weightSize);
            _bQDevice = new CudaDeviceVariable<float>(_modelDim);
            _bKDevice = new CudaDeviceVariable<float>(_modelDim);
            _bVDevice = new CudaDeviceVariable<float>(_modelDim);
            _bODevice = new CudaDeviceVariable<float>(_modelDim);

            // Copy weights to device
            CopyWeightsToDevice();

            // Create attention heads
            _attentionHeads = new ScaledDotProductAttentionCuda[_numHeads];
            for (int h = 0; h < _numHeads; h++)
            {
                _attentionHeads[h] = new ScaledDotProductAttentionCuda(_headDim, _useCausalMask, _dropout, _context);
                _attentionHeads[h].Training = _training;
            }

            Built = true;
        }

        private void CopyWeightsToDevice()
        {
            // Flatten 2D arrays for device copy
            float[] wQFlat = Flatten(_wQ);
            float[] wKFlat = Flatten(_wK);
            float[] wVFlat = Flatten(_wV);
            float[] wOFlat = Flatten(_wO);

            _wQDevice.CopyToDevice(wQFlat);
            _wKDevice.CopyToDevice(wKFlat);
            _wVDevice.CopyToDevice(wVFlat);
            _wODevice.CopyToDevice(wOFlat);
            _bQDevice.CopyToDevice(_bQ);
            _bKDevice.CopyToDevice(_bK);
            _bVDevice.CopyToDevice(_bV);
            _bODevice.CopyToDevice(_bO);
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
        public Tensor Forward(Tensor input, Tensor mask = null)
        {
            if (input.Rank != 3)
                throw new ArgumentException($"MultiHeadAttention expects rank-3 input, got {input.Rank}");

            int batch = input.Shape[0];
            int seqLen = input.Shape[1];

            if (!Built)
                Build(new[] { batch, seqLen, input.Shape[2] });

            _lastInput = input;

            // Project to Q, K, V using CUDA
            Tensor Q = LinearProjectionCuda(input, _wQDevice, _bQDevice, _modelDim);
            Tensor K = LinearProjectionCuda(input, _wKDevice, _bKDevice, _modelDim);
            Tensor V = LinearProjectionCuda(input, _wVDevice, _bVDevice, _modelDim);

            _lastQ = Q;
            _lastK = K;
            _lastV = V;

            // Split into heads and apply attention
            Tensor[] qHeads = SplitHeads(Q);
            Tensor[] kHeads = SplitHeads(K);
            Tensor[] vHeads = SplitHeads(V);

            Tensor[] headOutputs = new Tensor[_numHeads];

            // Process each head (could be parallelized further on GPU)
            for (int h = 0; h < _numHeads; h++)
            {
                headOutputs[h] = _attentionHeads[h].Forward(qHeads[h], kHeads[h], vHeads[h], mask);
            }

            // Concatenate heads
            Tensor concatenated = ConcatHeads(headOutputs);
            _lastAttnOutput = concatenated;

            // Output projection
            Tensor output = LinearProjectionCuda(concatenated, _wODevice, _bODevice, _modelDim);

            return output;
        }

        private Tensor LinearProjectionCuda(Tensor input, CudaDeviceVariable<float> weightsDevice,
            CudaDeviceVariable<float> biasDevice, int outputDim)
        {
            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int inputDim = input.Shape[2];

            int inputSize = batch * seqLen * inputDim;
            int outputSize = batch * seqLen * outputDim;

            using var inputDevice = new CudaDeviceVariable<float>(inputSize);
            using var outputDevice = new CudaDeviceVariable<float>(outputSize);

            inputDevice.CopyToDevice(input.Data);

            // Reshape for matrix multiplication: [batch*seqLen, inputDim] @ [inputDim, outputDim]
            var matmulKernel = _context.LoadKernelPTX(_denseKernelPath, "MatMul");
            dim3 blockSize = new dim3(16, 16);
            dim3 gridSize = new dim3(
                (uint)((outputDim + blockSize.x - 1) / blockSize.x),
                (uint)((batch * seqLen + blockSize.y - 1) / blockSize.y));

            matmulKernel.GridDimensions = gridSize;
            matmulKernel.BlockDimensions = blockSize;
            matmulKernel.Run(
                inputDevice.DevicePointer,
                weightsDevice.DevicePointer,
                outputDevice.DevicePointer,
                batch * seqLen, inputDim, outputDim);

            // Add bias - done on CPU for simplicity (could be fused into kernel)
            float[] outputData = new float[outputSize];
            outputDevice.CopyToHost(outputData);

            float[] biasData = new float[outputDim];
            biasDevice.CopyToHost(biasData);

            // Add bias to each position
            Parallel.For(0, batch * seqLen, row =>
            {
                int offset = row * outputDim;
                for (int o = 0; o < outputDim; o++)
                {
                    outputData[offset + o] += biasData[o];
                }
            });

            return new Tensor(outputData, new[] { batch, seqLen, outputDim });
        }

        /// <summary>
        /// Backward pass using Tensor API.
        /// </summary>
        public Tensor Backward(Tensor gradOutput)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];

            // Reset gradient accumulators
            ClearGradients();

            // Gradient through output projection
            Tensor dConcat = LinearProjectionBackward(gradOutput, _lastAttnOutput, _wO, _wOGrad, _bOGrad);

            // Split gradient for heads
            Tensor[] dConcatHeads = SplitHeads(dConcat);

            // Backward through each attention head
            Tensor[] dQHeads = new Tensor[_numHeads];
            Tensor[] dKHeads = new Tensor[_numHeads];
            Tensor[] dVHeads = new Tensor[_numHeads];

            Tensor[] qHeads = SplitHeads(_lastQ);
            Tensor[] kHeads = SplitHeads(_lastK);
            Tensor[] vHeads = SplitHeads(_lastV);

            for (int h = 0; h < _numHeads; h++)
            {
                // Re-run forward to cache values for this head
                _attentionHeads[h].Forward(qHeads[h], kHeads[h], vHeads[h], null);
                var (headDQ, headDK, headDV) = _attentionHeads[h].Backward(dConcatHeads[h]);
                dQHeads[h] = headDQ;
                dKHeads[h] = headDK;
                dVHeads[h] = headDV;
            }

            // Concatenate gradients from heads
            Tensor dQConcat = ConcatHeads(dQHeads);
            Tensor dKConcat = ConcatHeads(dKHeads);
            Tensor dVConcat = ConcatHeads(dVHeads);

            // Gradient through Q, K, V projections
            Tensor dInputFromQ = LinearProjectionBackward(dQConcat, _lastInput, _wQ, _wQGrad, _bQGrad);
            Tensor dInputFromK = LinearProjectionBackward(dKConcat, _lastInput, _wK, _wKGrad, _bKGrad);
            Tensor dInputFromV = LinearProjectionBackward(dVConcat, _lastInput, _wV, _wVGrad, _bVGrad);

            // Sum gradients from Q, K, V paths
            Tensor dInput = TensorOperations.Add(dInputFromQ, TensorOperations.Add(dInputFromK, dInputFromV));

            // Update device weights
            CopyWeightsToDevice();

            return dInput;
        }

        private const int PARALLEL_THRESHOLD = 64;

        private Tensor[] SplitHeads(Tensor t)
        {
            int batch = t.Shape[0];
            int seqLen = t.Shape[1];
            int totalRows = batch * seqLen;

            Tensor[] heads = new Tensor[_numHeads];
            for (int h = 0; h < _numHeads; h++)
            {
                heads[h] = new Tensor(new[] { batch, seqLen, _headDim });
            }

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, totalRows, row =>
                {
                    int b = row / seqLen;
                    int s = row % seqLen;
                    int inputOffset = (b * seqLen + s) * _modelDim;

                    for (int h = 0; h < _numHeads; h++)
                    {
                        int outputOffset = (b * seqLen + s) * _headDim;
                        int headOffset = h * _headDim;
                        for (int d = 0; d < _headDim; d++)
                        {
                            heads[h].Data[outputOffset + d] = t.Data[inputOffset + headOffset + d];
                        }
                    }
                });
            }
            else
            {
                for (int b = 0; b < batch; b++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int inputOffset = (b * seqLen + s) * _modelDim;
                        for (int h = 0; h < _numHeads; h++)
                        {
                            int outputOffset = (b * seqLen + s) * _headDim;
                            int headOffset = h * _headDim;
                            for (int d = 0; d < _headDim; d++)
                            {
                                heads[h].Data[outputOffset + d] = t.Data[inputOffset + headOffset + d];
                            }
                        }
                    }
                }
            }

            return heads;
        }

        private Tensor ConcatHeads(Tensor[] heads)
        {
            int batch = heads[0].Shape[0];
            int seqLen = heads[0].Shape[1];
            int totalRows = batch * seqLen;

            Tensor result = new Tensor(new[] { batch, seqLen, _modelDim });

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, totalRows, row =>
                {
                    int b = row / seqLen;
                    int s = row % seqLen;
                    int outputOffset = (b * seqLen + s) * _modelDim;

                    for (int h = 0; h < _numHeads; h++)
                    {
                        int inputOffset = (b * seqLen + s) * _headDim;
                        int headOffset = h * _headDim;
                        for (int d = 0; d < _headDim; d++)
                        {
                            result.Data[outputOffset + headOffset + d] = heads[h].Data[inputOffset + d];
                        }
                    }
                });
            }
            else
            {
                for (int b = 0; b < batch; b++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int outputOffset = (b * seqLen + s) * _modelDim;
                        for (int h = 0; h < _numHeads; h++)
                        {
                            int inputOffset = (b * seqLen + s) * _headDim;
                            int headOffset = h * _headDim;
                            for (int d = 0; d < _headDim; d++)
                            {
                                result.Data[outputOffset + headOffset + d] = heads[h].Data[inputOffset + d];
                            }
                        }
                    }
                }
            }

            return result;
        }

        private Tensor LinearProjectionBackward(Tensor gradOutput, Tensor input, float[,] weights,
                                                 float[,] weightGrad, float[] biasGrad)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];
            int outputDim = gradOutput.Shape[2];
            int inputDim = input.Shape[2];

            Tensor dInput = new Tensor(input.Shape);
            int totalRows = batch * seqLen;

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                var localWeightGrads = new System.Threading.ThreadLocal<float[,]>(
                    () => new float[inputDim, outputDim], trackAllValues: true);
                var localBiasGrads = new System.Threading.ThreadLocal<float[]>(
                    () => new float[outputDim], trackAllValues: true);

                Parallel.For(0, totalRows, row =>
                {
                    int b = row / seqLen;
                    int s = row % seqLen;
                    int inputOffset = (b * seqLen + s) * inputDim;
                    int gradOffset = (b * seqLen + s) * outputDim;

                    var localWGrad = localWeightGrads.Value;
                    var localBGrad = localBiasGrads.Value;

                    for (int o = 0; o < outputDim; o++)
                    {
                        localBGrad[o] += gradOutput.Data[gradOffset + o];
                    }

                    for (int i = 0; i < inputDim; i++)
                    {
                        float dInputSum = 0f;
                        float inputVal = input.Data[inputOffset + i];
                        for (int o = 0; o < outputDim; o++)
                        {
                            float gradVal = gradOutput.Data[gradOffset + o];
                            localWGrad[i, o] += inputVal * gradVal;
                            dInputSum += weights[i, o] * gradVal;
                        }
                        dInput.Data[inputOffset + i] = dInputSum;
                    }
                });

                foreach (var localWGrad in localWeightGrads.Values)
                {
                    for (int i = 0; i < inputDim; i++)
                    {
                        for (int o = 0; o < outputDim; o++)
                        {
                            weightGrad[i, o] += localWGrad[i, o];
                        }
                    }
                }

                foreach (var localBGrad in localBiasGrads.Values)
                {
                    for (int o = 0; o < outputDim; o++)
                    {
                        biasGrad[o] += localBGrad[o];
                    }
                }

                localWeightGrads.Dispose();
                localBiasGrads.Dispose();
            }
            else
            {
                for (int b = 0; b < batch; b++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int inputOffset = (b * seqLen + s) * inputDim;
                        int gradOffset = (b * seqLen + s) * outputDim;

                        for (int o = 0; o < outputDim; o++)
                        {
                            biasGrad[o] += gradOutput.Data[gradOffset + o];
                        }

                        for (int i = 0; i < inputDim; i++)
                        {
                            float dInputSum = 0f;
                            float inputVal = input.Data[inputOffset + i];
                            for (int o = 0; o < outputDim; o++)
                            {
                                float gradVal = gradOutput.Data[gradOffset + o];
                                weightGrad[i, o] += inputVal * gradVal;
                                dInputSum += weights[i, o] * gradVal;
                            }
                            dInput.Data[inputOffset + i] = dInputSum;
                        }
                    }
                }
            }

            return dInput;
        }

        private void ClearGradients()
        {
            Array.Clear(_wQGrad, 0, _wQGrad.Length);
            Array.Clear(_wKGrad, 0, _wKGrad.Length);
            Array.Clear(_wVGrad, 0, _wVGrad.Length);
            Array.Clear(_wOGrad, 0, _wOGrad.Length);
            Array.Clear(_bQGrad, 0, _bQGrad.Length);
            Array.Clear(_bKGrad, 0, _bKGrad.Length);
            Array.Clear(_bVGrad, 0, _bVGrad.Length);
            Array.Clear(_bOGrad, 0, _bOGrad.Length);
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
            throw new NotImplementedException("MultiHeadAttentionCuda does not support 4D tensors");
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException("MultiHeadAttentionCuda does not support 4D tensors");
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return inputShape;
        }

        public override void Dispose()
        {
            _wQDevice?.Dispose();
            _wKDevice?.Dispose();
            _wVDevice?.Dispose();
            _wODevice?.Dispose();
            _bQDevice?.Dispose();
            _bKDevice?.Dispose();
            _bVDevice?.Dispose();
            _bODevice?.Dispose();

            if (_attentionHeads != null)
            {
                foreach (var head in _attentionHeads)
                {
                    head?.Dispose();
                }
            }

            if (_contextOwned)
            {
                _context?.Dispose();
            }

            GC.SuppressFinalize(this);
        }
    }
}
