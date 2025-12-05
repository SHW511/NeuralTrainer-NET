using System;
using System.Threading.Tasks;
using NeuralNetwork.Tensors;
using NeuralNetwork.Initializers;

namespace NeuralNetwork.Layers.Attention
{
    /// <summary>
    /// Multi-Head Attention as described in "Attention Is All You Need" (Vaswani et al., 2017).
    ///
    /// MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
    /// where head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)
    /// </summary>
    public class MultiHeadAttention : Layer
    {
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _modelDim;
        private readonly float _dropout;
        private readonly bool _useCausalMask;

        // Projection weights
        private float[,] _wQ;  // [modelDim, modelDim]
        private float[,] _wK;  // [modelDim, modelDim]
        private float[,] _wV;  // [modelDim, modelDim]
        private float[,] _wO;  // [modelDim, modelDim]

        // Biases (optional but common)
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
        private ScaledDotProductAttention[] _attentionHeads;

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

        /// <summary>
        /// Create a Multi-Head Attention layer.
        /// </summary>
        /// <param name="modelDim">Model dimension (d_model).</param>
        /// <param name="numHeads">Number of attention heads.</param>
        /// <param name="dropout">Dropout rate for attention weights.</param>
        /// <param name="useCausalMask">Whether to apply causal masking.</param>
        public MultiHeadAttention(int modelDim, int numHeads, float dropout = 0f, bool useCausalMask = true)
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

            // Create attention heads
            _attentionHeads = new ScaledDotProductAttention[_numHeads];
            for (int h = 0; h < _numHeads; h++)
            {
                _attentionHeads[h] = new ScaledDotProductAttention(_headDim, _useCausalMask, _dropout);
                _attentionHeads[h].Training = _training;
            }

            Built = true;
        }

        /// <summary>
        /// Forward pass using Tensor API.
        /// </summary>
        /// <param name="input">Input tensor [batch, seqLen, modelDim].</param>
        /// <param name="mask">Optional additional mask.</param>
        /// <returns>Output tensor [batch, seqLen, modelDim].</returns>
        public Tensor Forward(Tensor input, Tensor mask = null)
        {
            if (input.Rank != 3)
                throw new ArgumentException($"MultiHeadAttention expects rank-3 input, got {input.Rank}");

            int batch = input.Shape[0];
            int seqLen = input.Shape[1];

            if (!Built)
                Build(new[] { batch, seqLen, input.Shape[2] });

            _lastInput = input;

            // Project to Q, K, V
            Tensor Q = LinearProjection(input, _wQ, _bQ);
            Tensor K = LinearProjection(input, _wK, _bK);
            Tensor V = LinearProjection(input, _wV, _bV);

            _lastQ = Q;
            _lastK = K;
            _lastV = V;

            // Split into heads: [batch, seqLen, modelDim] -> [batch, numHeads, seqLen, headDim]
            Tensor[] qHeads = SplitHeads(Q);
            Tensor[] kHeads = SplitHeads(K);
            Tensor[] vHeads = SplitHeads(V);

            // Apply attention per head
            Tensor[] headOutputs = new Tensor[_numHeads];
            for (int h = 0; h < _numHeads; h++)
            {
                headOutputs[h] = _attentionHeads[h].Forward(qHeads[h], kHeads[h], vHeads[h], mask);
            }

            // Concatenate heads: [batch, seqLen, modelDim]
            Tensor concatenated = ConcatHeads(headOutputs);
            _lastAttnOutput = concatenated;

            // Output projection
            Tensor output = LinearProjection(concatenated, _wO, _bO);

            return output;
        }

        /// <summary>
        /// Backward pass using Tensor API.
        /// </summary>
        /// <param name="gradOutput">Gradient w.r.t. output [batch, seqLen, modelDim].</param>
        /// <returns>Gradient w.r.t. input.</returns>
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
                // Need to re-run forward to cache values for this head
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

            return dInput;
        }

        private const int PARALLEL_THRESHOLD = 64;

        private Tensor LinearProjection(Tensor input, float[,] weights, float[] bias)
        {
            // input: [batch, seqLen, inputDim]
            // weights: [inputDim, outputDim]
            // output: [batch, seqLen, outputDim]

            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int inputDim = input.Shape[2];
            int outputDim = weights.GetLength(1);

            Tensor output = new Tensor(new[] { batch, seqLen, outputDim });
            int totalRows = batch * seqLen;

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, totalRows, row =>
                {
                    int b = row / seqLen;
                    int s = row % seqLen;
                    int inputOffset = (b * seqLen + s) * inputDim;
                    int outputOffset = (b * seqLen + s) * outputDim;

                    for (int o = 0; o < outputDim; o++)
                    {
                        float sum = bias[o];
                        for (int i = 0; i < inputDim; i++)
                        {
                            sum += input.Data[inputOffset + i] * weights[i, o];
                        }
                        output.Data[outputOffset + o] = sum;
                    }
                });
            }
            else
            {
                for (int b = 0; b < batch; b++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int inputOffset = (b * seqLen + s) * inputDim;
                        int outputOffset = (b * seqLen + s) * outputDim;

                        for (int o = 0; o < outputDim; o++)
                        {
                            float sum = bias[o];
                            for (int i = 0; i < inputDim; i++)
                            {
                                sum += input.Data[inputOffset + i] * weights[i, o];
                            }
                            output.Data[outputOffset + o] = sum;
                        }
                    }
                }
            }

            return output;
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
                // Thread-local gradient accumulators to avoid lock contention
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

                    // Bias gradient (thread-local)
                    for (int o = 0; o < outputDim; o++)
                    {
                        localBGrad[o] += gradOutput.Data[gradOffset + o];
                    }

                    // Weight gradient and input gradient
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

                // Aggregate thread-local gradients
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

                        // Bias gradient
                        for (int o = 0; o < outputDim; o++)
                        {
                            biasGrad[o] += gradOutput.Data[gradOffset + o];
                        }

                        // Weight gradient and input gradient
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

        private Tensor[] SplitHeads(Tensor t)
        {
            // t: [batch, seqLen, modelDim]
            // output: numHeads tensors of [batch, seqLen, headDim]

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
            // heads: numHeads tensors of [batch, seqLen, headDim]
            // output: [batch, seqLen, modelDim]

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
            // Convert to 3D tensor with batch=1
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

            // Convert back to 2D
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
            throw new NotImplementedException("MultiHeadAttention does not support 4D tensors");
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException("MultiHeadAttention does not support 4D tensors");
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return inputShape;
        }
    }
}
