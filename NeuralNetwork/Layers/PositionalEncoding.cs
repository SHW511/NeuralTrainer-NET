using System;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Layers
{
    /// <summary>
    /// Sinusoidal Positional Encoding as described in "Attention Is All You Need" (Vaswani et al., 2017).
    ///
    /// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    /// </summary>
    public class PositionalEncoding : Layer
    {
        private readonly int _maxSeqLen;
        private readonly int _modelDim;
        private readonly float _dropout;

        private float[,] _encodings;  // Precomputed encodings [maxSeqLen, modelDim]
        private Dropout _dropoutLayer;

        /// <summary>
        /// Create a Positional Encoding layer.
        /// </summary>
        /// <param name="maxSeqLen">Maximum sequence length to support.</param>
        /// <param name="modelDim">Model dimension (d_model).</param>
        /// <param name="dropout">Dropout rate to apply after adding positional encoding.</param>
        public PositionalEncoding(int maxSeqLen, int modelDim, float dropout = 0.1f)
        {
            _maxSeqLen = maxSeqLen;
            _modelDim = modelDim;
            _dropout = dropout;
            OutputDim = modelDim;

            ComputeEncodings();

            if (dropout > 0f)
            {
                _dropoutLayer = new Dropout(dropout);
            }
        }

        private void ComputeEncodings()
        {
            _encodings = new float[_maxSeqLen, _modelDim];

            for (int pos = 0; pos < _maxSeqLen; pos++)
            {
                for (int i = 0; i < _modelDim; i++)
                {
                    double angle = pos / Math.Pow(10000.0, (2.0 * (i / 2)) / _modelDim);

                    if (i % 2 == 0)
                    {
                        _encodings[pos, i] = (float)Math.Sin(angle);
                    }
                    else
                    {
                        _encodings[pos, i] = (float)Math.Cos(angle);
                    }
                }
            }
        }

        /// <summary>
        /// Get positional encoding for a given sequence length.
        /// </summary>
        /// <param name="seqLen">Sequence length.</param>
        /// <returns>Encoding tensor [seqLen, modelDim].</returns>
        public Tensor GetEncoding(int seqLen)
        {
            if (seqLen > _maxSeqLen)
                throw new ArgumentException($"Sequence length {seqLen} exceeds maximum {_maxSeqLen}");

            Tensor encoding = new Tensor(new[] { seqLen, _modelDim });

            for (int pos = 0; pos < seqLen; pos++)
            {
                for (int i = 0; i < _modelDim; i++)
                {
                    encoding[pos, i] = _encodings[pos, i];
                }
            }

            return encoding;
        }

        /// <summary>
        /// Get positional encoding for batched input.
        /// </summary>
        /// <param name="seqLen">Sequence length.</param>
        /// <param name="batchSize">Batch size (for broadcasting).</param>
        /// <returns>Encoding tensor [1, seqLen, modelDim] (broadcastable).</returns>
        public Tensor GetEncodingBatched(int seqLen, int batchSize = 1)
        {
            if (seqLen > _maxSeqLen)
                throw new ArgumentException($"Sequence length {seqLen} exceeds maximum {_maxSeqLen}");

            Tensor encoding = new Tensor(new[] { 1, seqLen, _modelDim });

            for (int pos = 0; pos < seqLen; pos++)
            {
                for (int i = 0; i < _modelDim; i++)
                {
                    encoding[0, pos, i] = _encodings[pos, i];
                }
            }

            return encoding;
        }

        public override void Build(int[] inputShape)
        {
            if (Built) return;

            InputShape = inputShape;
            InputDim = inputShape[inputShape.Length - 1];

            if (_dropoutLayer != null)
            {
                _dropoutLayer.Build(inputShape);
            }

            Built = true;
        }

        /// <summary>
        /// Forward pass: Add positional encoding to input embeddings.
        /// </summary>
        /// <param name="input">Input tensor [batch, seqLen, modelDim].</param>
        /// <returns>Output tensor with positional encoding added.</returns>
        public Tensor Forward(Tensor input)
        {
            if (input.Rank != 3)
                throw new ArgumentException($"PositionalEncoding expects rank-3 input, got {input.Rank}");

            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int dim = input.Shape[2];

            if (!Built)
                Build(new[] { batch, seqLen, dim });

            if (seqLen > _maxSeqLen)
                throw new ArgumentException($"Sequence length {seqLen} exceeds maximum {_maxSeqLen}");

            Tensor output = new Tensor(input.Shape);

            // Add positional encoding to each batch
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        output[b, s, d] = input[b, s, d] + _encodings[s, d];
                    }
                }
            }

            // Apply dropout if in training mode
            if (_dropoutLayer != null && _dropoutLayer.Training)
            {
                // Reshape for dropout
                float[,] output2D = new float[batch * seqLen, dim];
                for (int b = 0; b < batch; b++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            output2D[b * seqLen + s, d] = output[b, s, d];
                        }
                    }
                }

                float[,] dropped = _dropoutLayer.Call(output2D);

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
            }

            return output;
        }

        /// <summary>
        /// Backward pass. Positional encoding has no learnable parameters.
        /// </summary>
        public Tensor Backward(Tensor gradOutput)
        {
            // Gradient passes through unchanged (PE is fixed, not learned)
            if (_dropoutLayer != null && _dropoutLayer.Training)
            {
                int batch = gradOutput.Shape[0];
                int seqLen = gradOutput.Shape[1];
                int dim = gradOutput.Shape[2];

                // Apply dropout backward
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

                float[,] dInput2D = _dropoutLayer.Backward(grad2D);

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

            return gradOutput.Clone();
        }

        public override float[,] Call(float[,] inputs)
        {
            int seqLen = inputs.GetLength(0);
            int dim = inputs.GetLength(1);

            if (seqLen > _maxSeqLen)
                throw new ArgumentException($"Sequence length {seqLen} exceeds maximum {_maxSeqLen}");

            float[,] output = new float[seqLen, dim];

            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < dim; d++)
                {
                    output[s, d] = inputs[s, d] + _encodings[s, d];
                }
            }

            if (_dropoutLayer != null)
            {
                output = _dropoutLayer.Call(output);
            }

            return output;
        }

        public override float[,] Backward(float[,] gradient)
        {
            if (_dropoutLayer != null)
            {
                return _dropoutLayer.Backward(gradient);
            }
            return gradient;
        }

        public override float[,,,] Call(float[,,,] inputs)
        {
            throw new NotImplementedException("PositionalEncoding does not support 4D tensors");
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException("PositionalEncoding does not support 4D tensors");
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return inputShape;
        }

        /// <summary>
        /// Set training mode.
        /// </summary>
        public bool Training
        {
            get => _dropoutLayer?.Training ?? false;
            set
            {
                if (_dropoutLayer != null)
                    _dropoutLayer.Training = value;
            }
        }
    }
}
