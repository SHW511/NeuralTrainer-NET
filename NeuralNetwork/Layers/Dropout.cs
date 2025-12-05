using System;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Layers
{
    /// <summary>
    /// Dropout layer for regularization.
    /// Uses inverted dropout: scales outputs by 1/(1-p) during training.
    /// </summary>
    public class Dropout : Layer
    {
        private readonly float _rate;
        private readonly Random _rng;

        private float[,] _lastMask;
        private bool _training;

        /// <summary>
        /// Dropout rate (probability of dropping a unit).
        /// </summary>
        public float Rate => _rate;

        /// <summary>
        /// Whether the layer is in training mode.
        /// </summary>
        public bool Training
        {
            get => _training;
            set => _training = value;
        }

        /// <summary>
        /// Create a Dropout layer.
        /// </summary>
        /// <param name="rate">Dropout rate (0 to 1). E.g., 0.1 means 10% of units are dropped.</param>
        /// <param name="seed">Optional random seed for reproducibility.</param>
        public Dropout(float rate, int? seed = null)
        {
            if (rate < 0f || rate >= 1f)
                throw new ArgumentException("Dropout rate must be in [0, 1)");

            _rate = rate;
            _rng = seed.HasValue ? new Random(seed.Value) : new Random();
            _training = true;
        }

        public override void Build(int[] inputShape)
        {
            if (Built) return;

            InputShape = inputShape;
            if (inputShape.Length >= 1)
                InputDim = inputShape[inputShape.Length - 1];
            OutputDim = InputDim;

            Built = true;
        }

        public override float[,] Call(float[,] inputs)
        {
            int batchSize = inputs.GetLength(0);
            int features = inputs.GetLength(1);

            if (!Built)
                Build(new[] { batchSize, features });

            float[,] output = new float[batchSize, features];

            if (!_training || _rate == 0f)
            {
                // During inference, just pass through
                Array.Copy(inputs, output, inputs.Length);
                return output;
            }

            // During training, apply inverted dropout
            float scale = 1.0f / (1.0f - _rate);
            _lastMask = new float[batchSize, features];

            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    if (_rng.NextDouble() >= _rate)
                    {
                        _lastMask[b, f] = scale;
                        output[b, f] = inputs[b, f] * scale;
                    }
                    else
                    {
                        _lastMask[b, f] = 0f;
                        output[b, f] = 0f;
                    }
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
                throw new ArgumentException($"Dropout expects rank-2 input, got {input.Rank}");

            float[,] input2D = input.ToArray2D();
            float[,] output2D = Call(input2D);
            return Tensor.FromArray(output2D);
        }

        public override float[,] Backward(float[,] gradient)
        {
            int batchSize = gradient.GetLength(0);
            int features = gradient.GetLength(1);

            float[,] inputGrad = new float[batchSize, features];

            if (!_training || _rate == 0f || _lastMask == null)
            {
                // During inference, gradient passes through unchanged
                Array.Copy(gradient, inputGrad, gradient.Length);
                return inputGrad;
            }

            // During training, apply same mask
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    inputGrad[b, f] = gradient[b, f] * _lastMask[b, f];
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
                throw new ArgumentException($"Dropout backward expects rank-2 gradient, got {gradient.Rank}");

            float[,] grad2D = gradient.ToArray2D();
            float[,] inputGrad2D = Backward(grad2D);
            return Tensor.FromArray(inputGrad2D);
        }

        public override float[,,,] Call(float[,,,] inputs)
        {
            throw new NotImplementedException("Dropout does not support 4D tensors directly");
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException("Dropout does not support 4D tensors directly");
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return inputShape;
        }
    }
}
