using System;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Layers
{
    /// <summary>
    /// Layer Normalization as described in "Layer Normalization" (Ba et al., 2016).
    /// Normalizes across the last dimension (features) for each sample.
    /// </summary>
    public class LayerNorm : Layer
    {
        private readonly int _normalizedShape;
        private readonly float _epsilon;

        private float[] _gamma;  // Scale parameter
        private float[] _beta;   // Shift parameter

        // Cached values for backward pass
        private float[,] _lastInput;
        private float[] _lastMean;
        private float[] _lastVar;
        private float[,] _lastNormalized;

        // Gradient accumulators
        private float[] _gammaGrad;
        private float[] _betaGrad;

        public float[] Gamma => _gamma;
        public float[] Beta => _beta;
        public float[] GammaGrad => _gammaGrad;
        public float[] BetaGrad => _betaGrad;

        /// <summary>
        /// Create a LayerNorm layer.
        /// </summary>
        /// <param name="normalizedShape">Size of the last dimension to normalize over.</param>
        /// <param name="epsilon">Small constant for numerical stability.</param>
        public LayerNorm(int normalizedShape, float epsilon = 1e-5f)
        {
            _normalizedShape = normalizedShape;
            _epsilon = epsilon;
            OutputDim = normalizedShape;
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

            Built = true;
        }

        public override float[,] Call(float[,] inputs)
        {
            int batchSize = inputs.GetLength(0);
            int features = inputs.GetLength(1);

            if (!Built)
                Build(new[] { batchSize, features });

            _lastInput = inputs;
            _lastMean = new float[batchSize];
            _lastVar = new float[batchSize];
            _lastNormalized = new float[batchSize, features];

            float[,] output = new float[batchSize, features];

            for (int b = 0; b < batchSize; b++)
            {
                // Compute mean
                float mean = 0f;
                for (int f = 0; f < features; f++)
                {
                    mean += inputs[b, f];
                }
                mean /= features;
                _lastMean[b] = mean;

                // Compute variance
                float variance = 0f;
                for (int f = 0; f < features; f++)
                {
                    float diff = inputs[b, f] - mean;
                    variance += diff * diff;
                }
                variance /= features;
                _lastVar[b] = variance;

                // Normalize and apply scale/shift
                float stdInv = 1.0f / (float)Math.Sqrt(variance + _epsilon);
                for (int f = 0; f < features; f++)
                {
                    float normalized = (inputs[b, f] - mean) * stdInv;
                    _lastNormalized[b, f] = normalized;
                    output[b, f] = _gamma[f] * normalized + _beta[f];
                }
            }

            return output;
        }

        /// <summary>
        /// Forward pass using Tensor API.
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            if (input.Rank != 2)
                throw new ArgumentException($"LayerNorm expects rank-2 input, got {input.Rank}");

            float[,] input2D = input.ToArray2D();
            float[,] output2D = Call(input2D);
            return Tensor.FromArray(output2D);
        }

        public override float[,] Backward(float[,] gradient)
        {
            int batchSize = gradient.GetLength(0);
            int features = gradient.GetLength(1);

            float[,] inputGrad = new float[batchSize, features];

            // Reset gradient accumulators
            Array.Clear(_gammaGrad, 0, _gammaGrad.Length);
            Array.Clear(_betaGrad, 0, _betaGrad.Length);

            for (int b = 0; b < batchSize; b++)
            {
                float stdInv = 1.0f / (float)Math.Sqrt(_lastVar[b] + _epsilon);

                // Accumulate gradients for gamma and beta
                for (int f = 0; f < features; f++)
                {
                    _gammaGrad[f] += gradient[b, f] * _lastNormalized[b, f];
                    _betaGrad[f] += gradient[b, f];
                }

                // Compute gradient w.r.t. normalized input
                float[] dNorm = new float[features];
                for (int f = 0; f < features; f++)
                {
                    dNorm[f] = gradient[b, f] * _gamma[f];
                }

                // Compute gradient w.r.t. variance
                float dVar = 0f;
                for (int f = 0; f < features; f++)
                {
                    dVar += dNorm[f] * (_lastInput[b, f] - _lastMean[b]) * -0.5f *
                            (float)Math.Pow(_lastVar[b] + _epsilon, -1.5);
                }

                // Compute gradient w.r.t. mean
                float dMean = 0f;
                for (int f = 0; f < features; f++)
                {
                    dMean += dNorm[f] * -stdInv;
                }
                dMean += dVar * -2f / features * SumDiff(b, features);

                // Compute gradient w.r.t. input
                for (int f = 0; f < features; f++)
                {
                    inputGrad[b, f] = dNorm[f] * stdInv +
                                      dVar * 2f * (_lastInput[b, f] - _lastMean[b]) / features +
                                      dMean / features;
                }
            }

            return inputGrad;
        }

        /// <summary>
        /// Backward pass using Tensor API.
        /// </summary>
        public Tensor Backward(Tensor gradient)
        {
            if (gradient.Rank != 2)
                throw new ArgumentException($"LayerNorm backward expects rank-2 gradient, got {gradient.Rank}");

            float[,] grad2D = gradient.ToArray2D();
            float[,] inputGrad2D = Backward(grad2D);
            return Tensor.FromArray(inputGrad2D);
        }

        private float SumDiff(int batch, int features)
        {
            float sum = 0f;
            for (int f = 0; f < features; f++)
            {
                sum += _lastInput[batch, f] - _lastMean[batch];
            }
            return sum;
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
        }

        public override float[,,,] Call(float[,,,] inputs)
        {
            throw new NotImplementedException("LayerNorm does not support 4D tensors directly");
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException("LayerNorm does not support 4D tensors directly");
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return inputShape;
        }
    }
}
