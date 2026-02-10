using System;
using System.Collections.Generic;
using System.IO;
using ManagedCuda;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Layers.Cuda
{
    /// <summary>
    /// CUDA-accelerated Transformer Block (decoder-only, pre-LayerNorm).
    ///
    /// Architecture:
    /// x = x + MultiHeadAttention(LayerNorm(x))
    /// x = x + FeedForward(LayerNorm(x))
    /// </summary>
    public class TransformerBlockCuda : Layer
    {
        private readonly int _modelDim;
        private readonly int _numHeads;
        private readonly int _ffDim;
        private readonly float _dropout;
        private readonly bool _useCausalMask;

        private CudaContext _context;
        private bool _contextOwned;
        private string _kernelPath;

        // Sub-layers
        private LayerNormCuda _attnNorm;
        private MultiHeadAttentionCuda _attention;
        private LayerNormCuda _ffnNorm;
        private FeedForwardCuda _feedForward;

        // Dropout layers (using CPU for simplicity, can be moved to GPU)
        private Dropout _attnDropout;
        private Dropout _ffnDropout;

        // Cached values for backward pass
        private Tensor _lastInput;
        private Tensor _lastAttnNormOutput;
        private Tensor _lastAttnOutput;
        private Tensor _lastAfterAttnResidual;
        private Tensor _lastFfnNormOutput;
        private Tensor _lastFfnOutput;

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
                if (_ffnDropout != null) _ffnDropout.Training = value;
            }
        }

        /// <summary>
        /// Create a CUDA-accelerated Transformer Block.
        /// </summary>
        /// <param name="modelDim">Model dimension (d_model).</param>
        /// <param name="numHeads">Number of attention heads.</param>
        /// <param name="ffDim">Feed-forward dimension (d_ff). Typically 4 * d_model.</param>
        /// <param name="dropout">Dropout rate.</param>
        /// <param name="useCausalMask">Whether to apply causal masking in attention.</param>
        /// <param name="context">Optional shared CUDA context.</param>
        public TransformerBlockCuda(int modelDim, int numHeads, int ffDim, float dropout = 0.1f,
            bool useCausalMask = true, CudaContext context = null)
        {
            _modelDim = modelDim;
            _numHeads = numHeads;
            _ffDim = ffDim;
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

            // Initialize sub-layers with shared context
            _attnNorm = new LayerNormCuda(modelDim, context: _context);
            _attention = new MultiHeadAttentionCuda(modelDim, numHeads, dropout, useCausalMask, _context);
            _ffnNorm = new LayerNormCuda(modelDim, context: _context);
            _feedForward = new FeedForwardCuda(modelDim, ffDim, dropout, _context);

            if (dropout > 0f)
            {
                _attnDropout = new Dropout(dropout);
                _ffnDropout = new Dropout(dropout);
            }
        }

        public override void Build(int[] inputShape)
        {
            if (Built) return;

            InputShape = inputShape;
            InputDim = inputShape[inputShape.Length - 1];

            if (InputDim != _modelDim)
                throw new ArgumentException($"Input dimension {InputDim} doesn't match model dimension {_modelDim}");

            // Build sub-layers
            _attnNorm.Build(inputShape);
            _attention.Build(inputShape);
            _ffnNorm.Build(inputShape);
            _feedForward.Build(inputShape);

            if (_attnDropout != null)
            {
                _attnDropout.Build(inputShape);
            }
            if (_ffnDropout != null)
            {
                _ffnDropout.Build(inputShape);
            }

            Built = true;
        }

        /// <summary>
        /// Forward pass using Tensor API with CUDA acceleration.
        /// </summary>
        public Tensor Forward(Tensor input, Tensor mask = null)
        {
            if (input.Rank != 3)
                throw new ArgumentException($"TransformerBlock expects rank-3 input, got {input.Rank}");

            int batch = input.Shape[0];
            int seqLen = input.Shape[1];

            if (!Built)
                Build(new[] { batch, seqLen, input.Shape[2] });

            _lastInput = input;

            // Attention sub-layer with pre-norm and residual
            // x = x + Dropout(Attention(LayerNorm(x)))
            Tensor attnNormOutput = _attnNorm.Forward(input);
            _lastAttnNormOutput = attnNormOutput;

            Tensor attnOutput = _attention.Forward(attnNormOutput, mask);
            _lastAttnOutput = attnOutput;

            // Apply dropout to attention output
            if (_training && _attnDropout != null)
            {
                attnOutput = ApplyDropout3D(attnOutput, _attnDropout);
            }

            // Residual connection
            Tensor afterAttnResidual = ResidualAddCuda(input, attnOutput);
            _lastAfterAttnResidual = afterAttnResidual;

            // Feed-forward sub-layer with pre-norm and residual
            // x = x + Dropout(FFN(LayerNorm(x)))
            Tensor ffnNormOutput = _ffnNorm.Forward(afterAttnResidual);
            _lastFfnNormOutput = ffnNormOutput;

            Tensor ffnOutput = _feedForward.Forward(ffnNormOutput);
            _lastFfnOutput = ffnOutput;

            // Apply dropout to FFN output
            if (_training && _ffnDropout != null)
            {
                ffnOutput = ApplyDropout3D(ffnOutput, _ffnDropout);
            }

            // Residual connection
            Tensor output = ResidualAddCuda(afterAttnResidual, ffnOutput);

            return output;
        }

        /// <summary>
        /// Backward pass using Tensor API.
        /// </summary>
        public Tensor Backward(Tensor gradOutput)
        {
            // Backward through second residual
            // d_afterAttnResidual = gradOutput
            // d_ffnOutput = gradOutput (dropout backward applied if training)
            Tensor dFfnOutput = gradOutput;
            if (_training && _ffnDropout != null)
            {
                dFfnOutput = ApplyDropoutBackward3D(dFfnOutput, _ffnDropout);
            }

            // Backward through FFN
            Tensor dFfnNormOutput = _feedForward.Backward(dFfnOutput);

            // Backward through FFN LayerNorm
            Tensor dAfterAttnResidual = _ffnNorm.Backward(dFfnNormOutput);

            // Add residual gradient
            dAfterAttnResidual = TensorOperations.Add(dAfterAttnResidual, gradOutput);

            // Backward through attention dropout
            Tensor dAttnOutput = dAfterAttnResidual;
            if (_training && _attnDropout != null)
            {
                dAttnOutput = ApplyDropoutBackward3D(dAttnOutput, _attnDropout);
            }

            // Backward through attention
            Tensor dAttnNormOutput = _attention.Backward(dAttnOutput);

            // Backward through attention LayerNorm
            Tensor dInput = _attnNorm.Backward(dAttnNormOutput);

            // Add residual gradient from first residual
            dInput = TensorOperations.Add(dInput, dAfterAttnResidual);

            return dInput;
        }

        private Tensor ResidualAddCuda(Tensor a, Tensor b)
        {
            // Use TensorOperations for now, could be moved to GPU kernel
            return TensorOperations.Add(a, b);
        }

        private Tensor ApplyDropout3D(Tensor input, Dropout dropout)
        {
            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int dim = input.Shape[2];
            int totalRows = batch * seqLen;

            // Reshape to 2D for dropout
            float[,] input2D = new float[totalRows, dim];
            for (int row = 0; row < totalRows; row++)
            {
                int offset = row * dim;
                for (int d = 0; d < dim; d++)
                {
                    input2D[row, d] = input.Data[offset + d];
                }
            }

            float[,] dropped = dropout.Call(input2D);

            Tensor output = new Tensor(input.Shape);
            for (int row = 0; row < totalRows; row++)
            {
                int offset = row * dim;
                for (int d = 0; d < dim; d++)
                {
                    output.Data[offset + d] = dropped[row, d];
                }
            }

            return output;
        }

        private Tensor ApplyDropoutBackward3D(Tensor gradOutput, Dropout dropout)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];
            int dim = gradOutput.Shape[2];
            int totalRows = batch * seqLen;

            float[,] grad2D = new float[totalRows, dim];
            for (int row = 0; row < totalRows; row++)
            {
                int offset = row * dim;
                for (int d = 0; d < dim; d++)
                {
                    grad2D[row, d] = gradOutput.Data[offset + d];
                }
            }

            float[,] dInput2D = dropout.Backward(grad2D);

            Tensor dInput = new Tensor(gradOutput.Shape);
            for (int row = 0; row < totalRows; row++)
            {
                int offset = row * dim;
                for (int d = 0; d < dim; d++)
                {
                    dInput.Data[offset + d] = dInput2D[row, d];
                }
            }

            return dInput;
        }

        /// <summary>
        /// Get all trainable parameters for optimizer.
        /// </summary>
        public (List<float[,]> weights, List<float[]> biases, List<float[,]> weightGrads, List<float[]> biasGrads) GetParameters()
        {
            var weights = new List<float[,]>
            {
                _attention.WQ, _attention.WK, _attention.WV, _attention.WO,
                _feedForward.W1, _feedForward.W2
            };

            var biases = new List<float[]>
            {
                _attnNorm.Gamma, _attnNorm.Beta,
                _ffnNorm.Gamma, _ffnNorm.Beta,
                _feedForward.B1, _feedForward.B2
            };

            var weightGrads = new List<float[,]>
            {
                _attention.WQGrad, _attention.WKGrad, _attention.WVGrad, _attention.WOGrad,
                _feedForward.W1Grad, _feedForward.W2Grad
            };

            var biasGrads = new List<float[]>
            {
                _attnNorm.GammaGrad, _attnNorm.BetaGrad,
                _ffnNorm.GammaGrad, _ffnNorm.BetaGrad,
                _feedForward.B1Grad, _feedForward.B2Grad
            };

            return (weights, biases, weightGrads, biasGrads);
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
            throw new NotImplementedException("TransformerBlockCuda does not support 4D tensors");
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException("TransformerBlockCuda does not support 4D tensors");
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return inputShape;
        }

        public override void Dispose()
        {
            _attnNorm?.Dispose();
            _attention?.Dispose();
            _ffnNorm?.Dispose();
            _feedForward?.Dispose();

            // Dispose dropout layers via base class Dispose
            _attnDropout?.Dispose();
            _ffnDropout?.Dispose();

            if (_contextOwned)
            {
                _context?.Dispose();
            }

            GC.SuppressFinalize(this);
        }
    }
}
