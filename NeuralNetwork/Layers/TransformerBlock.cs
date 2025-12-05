using System;
using NeuralNetwork.Tensors;
using NeuralNetwork.Layers.Attention;

namespace NeuralNetwork.Layers
{
    /// <summary>
    /// Transformer Decoder Block with Pre-LN architecture (more stable training).
    ///
    /// Block structure:
    /// x_norm = LayerNorm(x)
    /// x = x + Dropout(MultiHeadAttention(x_norm))
    /// x_norm = LayerNorm(x)
    /// x = x + Dropout(FeedForward(x_norm))
    /// </summary>
    public class TransformerBlock : Layer
    {
        private readonly int _modelDim;
        private readonly int _numHeads;
        private readonly int _ffDim;
        private readonly float _dropout;
        private readonly bool _useCausalMask;

        // Sub-layers
        private LayerNorm _attnLayerNorm;
        private MultiHeadAttention _attention;
        private Dropout _attnDropout;
        private LayerNorm _ffLayerNorm;
        private FeedForward _feedForward;
        private Dropout _ffDropout;

        // Cached values for backward pass
        private Tensor _lastInput;
        private Tensor _lastAttnNormOutput;
        private Tensor _lastAttnOutput;
        private Tensor _lastAfterAttnResidual;
        private Tensor _lastFFNormOutput;
        private Tensor _lastFFOutput;

        private bool _training;

        public bool Training
        {
            get => _training;
            set
            {
                _training = value;
                _attention.Training = value;
                _feedForward.Training = value;
                if (_attnDropout != null) _attnDropout.Training = value;
                if (_ffDropout != null) _ffDropout.Training = value;
            }
        }

        // Expose sub-layers for parameter access
        public LayerNorm AttnLayerNorm => _attnLayerNorm;
        public MultiHeadAttention Attention => _attention;
        public LayerNorm FFLayerNorm => _ffLayerNorm;
        public FeedForward FeedForwardLayer => _feedForward;

        /// <summary>
        /// Create a Transformer Block.
        /// </summary>
        /// <param name="modelDim">Model dimension (d_model).</param>
        /// <param name="numHeads">Number of attention heads.</param>
        /// <param name="ffDim">Feed-forward dimension (d_ff). Typically 4 * d_model.</param>
        /// <param name="dropout">Dropout rate.</param>
        /// <param name="useCausalMask">Whether to use causal masking for autoregressive generation.</param>
        public TransformerBlock(int modelDim, int numHeads, int ffDim, float dropout = 0.1f, bool useCausalMask = true)
        {
            _modelDim = modelDim;
            _numHeads = numHeads;
            _ffDim = ffDim;
            _dropout = dropout;
            _useCausalMask = useCausalMask;
            _training = true;
            OutputDim = modelDim;

            // Initialize sub-layers
            _attnLayerNorm = new LayerNorm(modelDim);
            _attention = new MultiHeadAttention(modelDim, numHeads, dropout: 0f, useCausalMask: useCausalMask);
            _ffLayerNorm = new LayerNorm(modelDim);
            _feedForward = new FeedForward(modelDim, ffDim, dropout: 0f);

            // Residual dropout
            if (dropout > 0f)
            {
                _attnDropout = new Dropout(dropout);
                _ffDropout = new Dropout(dropout);
            }
        }

        public override void Build(int[] inputShape)
        {
            if (Built) return;

            InputShape = inputShape;
            InputDim = inputShape[inputShape.Length - 1];

            _attnLayerNorm.Build(new[] { inputShape[1], inputShape[2] });
            _attention.Build(inputShape);
            _ffLayerNorm.Build(new[] { inputShape[1], inputShape[2] });
            _feedForward.Build(inputShape);

            if (_attnDropout != null)
                _attnDropout.Build(new[] { inputShape[0] * inputShape[1], inputShape[2] });
            if (_ffDropout != null)
                _ffDropout.Build(new[] { inputShape[0] * inputShape[1], inputShape[2] });

            Built = true;
        }

        /// <summary>
        /// Forward pass using Tensor API.
        /// </summary>
        /// <param name="input">Input tensor [batch, seqLen, modelDim].</param>
        /// <param name="mask">Optional additional attention mask.</param>
        /// <returns>Output tensor [batch, seqLen, modelDim].</returns>
        public Tensor Forward(Tensor input, Tensor mask = null)
        {
            if (input.Rank != 3)
                throw new ArgumentException($"TransformerBlock expects rank-3 input, got {input.Rank}");

            int batch = input.Shape[0];
            int seqLen = input.Shape[1];

            if (!Built)
                Build(new[] { batch, seqLen, input.Shape[2] });

            _lastInput = input;

            // Pre-LN: LayerNorm before attention
            Tensor attnNormOutput = ApplyLayerNorm3D(input, _attnLayerNorm);
            _lastAttnNormOutput = attnNormOutput;

            // Multi-Head Attention
            Tensor attnOutput = _attention.Forward(attnNormOutput, mask);
            _lastAttnOutput = attnOutput;

            // Dropout on attention output
            if (_attnDropout != null && _training)
            {
                attnOutput = ApplyDropout3D(attnOutput, _attnDropout);
            }

            // Residual connection
            Tensor afterAttnResidual = TensorOperations.Add(input, attnOutput);
            _lastAfterAttnResidual = afterAttnResidual;

            // Pre-LN: LayerNorm before FFN
            Tensor ffNormOutput = ApplyLayerNorm3D(afterAttnResidual, _ffLayerNorm);
            _lastFFNormOutput = ffNormOutput;

            // Feed-Forward Network
            Tensor ffOutput = _feedForward.Forward(ffNormOutput);
            _lastFFOutput = ffOutput;

            // Dropout on FFN output
            if (_ffDropout != null && _training)
            {
                ffOutput = ApplyDropout3D(ffOutput, _ffDropout);
            }

            // Residual connection
            Tensor output = TensorOperations.Add(afterAttnResidual, ffOutput);

            return output;
        }

        /// <summary>
        /// Backward pass using Tensor API.
        /// </summary>
        public Tensor Backward(Tensor gradOutput)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];

            // Gradient through FFN residual: splits to both paths
            Tensor dFFOutput = gradOutput.Clone();
            Tensor dAfterAttnResidual = gradOutput.Clone();

            // Gradient through FFN dropout
            if (_ffDropout != null && _training)
            {
                dFFOutput = ApplyDropoutBackward3D(dFFOutput, _ffDropout);
            }

            // Gradient through FFN
            Tensor dFFNormOutput = _feedForward.Backward(dFFOutput);

            // Gradient through FFN LayerNorm
            Tensor dFromFF = ApplyLayerNormBackward3D(dFFNormOutput, _ffLayerNorm, _lastAfterAttnResidual);

            // Add gradient from FFN path to residual gradient
            dAfterAttnResidual = TensorOperations.Add(dAfterAttnResidual, dFromFF);

            // Gradient through attention residual: splits to both paths
            Tensor dAttnOutput = dAfterAttnResidual.Clone();
            Tensor dInput = dAfterAttnResidual.Clone();

            // Gradient through attention dropout
            if (_attnDropout != null && _training)
            {
                dAttnOutput = ApplyDropoutBackward3D(dAttnOutput, _attnDropout);
            }

            // Gradient through Multi-Head Attention
            Tensor dAttnNormOutput = _attention.Backward(dAttnOutput);

            // Gradient through attention LayerNorm
            Tensor dFromAttn = ApplyLayerNormBackward3D(dAttnNormOutput, _attnLayerNorm, _lastInput);

            // Add gradient from attention path to input gradient
            dInput = TensorOperations.Add(dInput, dFromAttn);

            return dInput;
        }

        private Tensor ApplyLayerNorm3D(Tensor input, LayerNorm layerNorm)
        {
            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int dim = input.Shape[2];

            // Reshape to 2D
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

            float[,] output2D = layerNorm.Call(input2D);

            // Reshape back to 3D
            Tensor output = new Tensor(input.Shape);
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        output[b, s, d] = output2D[b * seqLen + s, d];
                    }
                }
            }

            return output;
        }

        private Tensor ApplyLayerNormBackward3D(Tensor gradOutput, LayerNorm layerNorm, Tensor originalInput)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];
            int dim = gradOutput.Shape[2];

            // Need to re-run forward to set up cached values
            float[,] input2D = new float[batch * seqLen, dim];
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        input2D[b * seqLen + s, d] = originalInput[b, s, d];
                    }
                }
            }
            layerNorm.Call(input2D);

            // Now do backward
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

            float[,] dInput2D = layerNorm.Backward(grad2D);

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
            throw new NotImplementedException("TransformerBlock does not support 4D tensors");
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException("TransformerBlock does not support 4D tensors");
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return inputShape;
        }
    }
}
