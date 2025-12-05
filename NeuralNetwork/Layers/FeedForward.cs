using System;
using System.Threading.Tasks;
using NeuralNetwork.Tensors;
using NeuralNetwork.Initializers;
using NeuralNetwork.Layers.Activations;

namespace NeuralNetwork.Layers
{
    /// <summary>
    /// Position-wise Feed-Forward Network as used in Transformers.
    ///
    /// FFN(x) = Linear2(GELU(Linear1(x)))
    ///
    /// Linear1: d_model -> d_ff (expansion)
    /// Linear2: d_ff -> d_model (projection)
    /// </summary>
    public class FeedForward : Layer
    {
        private readonly int _modelDim;
        private readonly int _ffDim;
        private readonly float _dropout;

        // Weights and biases
        private float[,] _w1;  // [modelDim, ffDim]
        private float[] _b1;   // [ffDim]
        private float[,] _w2;  // [ffDim, modelDim]
        private float[] _b2;   // [modelDim]

        // Gradient accumulators
        private float[,] _w1Grad;
        private float[] _b1Grad;
        private float[,] _w2Grad;
        private float[] _b2Grad;

        // Cached values for backward pass
        private Tensor _lastInput;
        private Tensor _lastHidden;      // After first linear + GELU
        private Tensor _lastPreGelu;     // Before GELU (for derivative)
        private Dropout _dropoutLayer;

        private bool _training;

        public bool Training
        {
            get => _training;
            set
            {
                _training = value;
                if (_dropoutLayer != null)
                    _dropoutLayer.Training = value;
            }
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
        /// Create a Feed-Forward Network layer.
        /// </summary>
        /// <param name="modelDim">Model dimension (d_model).</param>
        /// <param name="ffDim">Feed-forward dimension (d_ff). Typically 4 * d_model.</param>
        /// <param name="dropout">Dropout rate.</param>
        public FeedForward(int modelDim, int ffDim, float dropout = 0.1f)
        {
            _modelDim = modelDim;
            _ffDim = ffDim;
            _dropout = dropout;
            _training = true;
            OutputDim = modelDim;

            if (dropout > 0f)
            {
                _dropoutLayer = new Dropout(dropout);
            }
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

            if (_dropoutLayer != null)
            {
                _dropoutLayer.Build(new[] { inputShape[0], inputShape[1], _ffDim });
            }

            Built = true;
        }

        /// <summary>
        /// Forward pass using Tensor API.
        /// </summary>
        /// <param name="input">Input tensor [batch, seqLen, modelDim].</param>
        /// <returns>Output tensor [batch, seqLen, modelDim].</returns>
        public Tensor Forward(Tensor input)
        {
            if (input.Rank != 3)
                throw new ArgumentException($"FeedForward expects rank-3 input, got {input.Rank}");

            int batch = input.Shape[0];
            int seqLen = input.Shape[1];

            if (!Built)
                Build(new[] { batch, seqLen, input.Shape[2] });

            _lastInput = input;

            // First linear: [batch, seqLen, modelDim] -> [batch, seqLen, ffDim]
            Tensor preGelu = LinearForward(input, _w1, _b1);
            _lastPreGelu = preGelu;

            // GELU activation
            Tensor hidden = GELU.Apply(preGelu);

            // Dropout
            if (_dropoutLayer != null && _training)
            {
                hidden = ApplyDropout3D(hidden);
            }

            _lastHidden = hidden;

            // Second linear: [batch, seqLen, ffDim] -> [batch, seqLen, modelDim]
            Tensor output = LinearForward(hidden, _w2, _b2);

            return output;
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

            // Gradient through second linear
            Tensor dHidden = LinearBackward(gradOutput, _lastHidden, _w2, _w2Grad, _b2Grad);

            // Gradient through dropout
            if (_dropoutLayer != null && _training)
            {
                dHidden = ApplyDropoutBackward3D(dHidden);
            }

            // Gradient through GELU
            Tensor dPreGelu = GELU.Backward(_lastPreGelu, dHidden);

            // Gradient through first linear
            Tensor dInput = LinearBackward(dPreGelu, _lastInput, _w1, _w1Grad, _b1Grad);

            return dInput;
        }

        private const int PARALLEL_THRESHOLD = 64;

        private Tensor LinearForward(Tensor input, float[,] weights, float[] bias)
        {
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

        private Tensor LinearBackward(Tensor gradOutput, Tensor input, float[,] weights,
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

        private Tensor ApplyDropout3D(Tensor input)
        {
            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int dim = input.Shape[2];
            int totalRows = batch * seqLen;

            // Reshape to 2D for dropout
            float[,] input2D = new float[totalRows, dim];

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, totalRows, row =>
                {
                    int srcOffset = row * dim;
                    for (int d = 0; d < dim; d++)
                    {
                        input2D[row, d] = input.Data[srcOffset + d];
                    }
                });
            }
            else
            {
                for (int row = 0; row < totalRows; row++)
                {
                    int srcOffset = row * dim;
                    for (int d = 0; d < dim; d++)
                    {
                        input2D[row, d] = input.Data[srcOffset + d];
                    }
                }
            }

            float[,] dropped = _dropoutLayer.Call(input2D);

            Tensor output = new Tensor(input.Shape);

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, totalRows, row =>
                {
                    int dstOffset = row * dim;
                    for (int d = 0; d < dim; d++)
                    {
                        output.Data[dstOffset + d] = dropped[row, d];
                    }
                });
            }
            else
            {
                for (int row = 0; row < totalRows; row++)
                {
                    int dstOffset = row * dim;
                    for (int d = 0; d < dim; d++)
                    {
                        output.Data[dstOffset + d] = dropped[row, d];
                    }
                }
            }

            return output;
        }

        private Tensor ApplyDropoutBackward3D(Tensor gradOutput)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];
            int dim = gradOutput.Shape[2];
            int totalRows = batch * seqLen;

            float[,] grad2D = new float[totalRows, dim];

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, totalRows, row =>
                {
                    int srcOffset = row * dim;
                    for (int d = 0; d < dim; d++)
                    {
                        grad2D[row, d] = gradOutput.Data[srcOffset + d];
                    }
                });
            }
            else
            {
                for (int row = 0; row < totalRows; row++)
                {
                    int srcOffset = row * dim;
                    for (int d = 0; d < dim; d++)
                    {
                        grad2D[row, d] = gradOutput.Data[srcOffset + d];
                    }
                }
            }

            float[,] dInput2D = _dropoutLayer.Backward(grad2D);

            Tensor dInput = new Tensor(gradOutput.Shape);

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, totalRows, row =>
                {
                    int dstOffset = row * dim;
                    for (int d = 0; d < dim; d++)
                    {
                        dInput.Data[dstOffset + d] = dInput2D[row, d];
                    }
                });
            }
            else
            {
                for (int row = 0; row < totalRows; row++)
                {
                    int dstOffset = row * dim;
                    for (int d = 0; d < dim; d++)
                    {
                        dInput.Data[dstOffset + d] = dInput2D[row, d];
                    }
                }
            }

            return dInput;
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
            throw new NotImplementedException("FeedForward does not support 4D tensors");
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException("FeedForward does not support 4D tensors");
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return inputShape;
        }
    }
}
