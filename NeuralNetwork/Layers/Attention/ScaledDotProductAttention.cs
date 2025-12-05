using System;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Layers.Attention
{
    /// <summary>
    /// Scaled Dot-Product Attention as described in "Attention Is All You Need" (Vaswani et al., 2017).
    ///
    /// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    /// </summary>
    public class ScaledDotProductAttention
    {
        private readonly float _scale;
        private readonly bool _useCausalMask;
        private readonly float _dropout;
        private readonly Random _rng;

        // Cached values for backward pass
        private Tensor _lastQ;
        private Tensor _lastK;
        private Tensor _lastV;
        private Tensor _lastScores;
        private Tensor _lastAttnWeights;
        private Tensor _lastDropoutMask;
        private bool _training;

        /// <summary>
        /// Create a Scaled Dot-Product Attention module.
        /// </summary>
        /// <param name="headDim">Dimension of each attention head (d_k).</param>
        /// <param name="useCausalMask">Whether to apply causal masking.</param>
        /// <param name="dropout">Dropout rate for attention weights.</param>
        public ScaledDotProductAttention(int headDim, bool useCausalMask = true, float dropout = 0f)
        {
            _scale = 1.0f / (float)Math.Sqrt(headDim);
            _useCausalMask = useCausalMask;
            _dropout = dropout;
            _rng = new Random();
            _training = true;
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
        /// Forward pass.
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

            // Compute attention scores: QK^T
            // Q: [batch, seqLen, headDim]
            // K^T: [batch, headDim, seqLen]
            // Scores: [batch, seqLen, seqLen]
            Tensor keyT = TransposeLast2D(key);
            Tensor scores = TensorOperations.BatchedMatMul(query, keyT);

            // Scale
            scores = TensorOperations.Scale(scores, _scale);
            _lastScores = scores;

            // Apply causal mask if needed
            if (_useCausalMask)
            {
                Tensor causalMask = AttentionMask.CreateCausalMaskBatched(seqLen);
                scores = BroadcastAdd(scores, causalMask);
            }

            // Apply additional mask if provided
            if (mask != null)
            {
                scores = TensorOperations.Add(scores, mask);
            }

            // Softmax along last dimension
            Tensor attnWeights = Softmax3D(scores);
            _lastAttnWeights = attnWeights;

            // Apply dropout to attention weights
            if (_training && _dropout > 0f)
            {
                attnWeights = ApplyDropout(attnWeights);
            }

            // Compute output: attnWeights @ V
            // attnWeights: [batch, seqLen, seqLen]
            // V: [batch, seqLen, headDim]
            // Output: [batch, seqLen, headDim]
            Tensor output = TensorOperations.BatchedMatMul(attnWeights, value);

            return output;
        }

        /// <summary>
        /// Backward pass.
        /// </summary>
        /// <param name="gradOutput">Gradient w.r.t. output [batch, seqLen, headDim].</param>
        /// <returns>Gradients w.r.t. Q, K, V.</returns>
        public (Tensor dQ, Tensor dK, Tensor dV) Backward(Tensor gradOutput)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];
            int headDim = gradOutput.Shape[2];

            // Gradient w.r.t. V: attnWeights^T @ gradOutput
            // attnWeights^T: [batch, seqLen, seqLen] -> transpose last 2 dims
            Tensor attnWeightsT = TransposeLast2D(_lastAttnWeights);
            Tensor dV = TensorOperations.BatchedMatMul(attnWeightsT, gradOutput);

            // Gradient w.r.t. attnWeights: gradOutput @ V^T
            Tensor valueT = TransposeLast2D(_lastV);
            Tensor dAttnWeights = TensorOperations.BatchedMatMul(gradOutput, valueT);

            // Apply dropout gradient
            if (_training && _dropout > 0f && _lastDropoutMask != null)
            {
                dAttnWeights = TensorOperations.Multiply(dAttnWeights, _lastDropoutMask);
            }

            // Softmax backward: d_scores = attnWeights * (d_attnWeights - sum(d_attnWeights * attnWeights))
            Tensor dScores = SoftmaxBackward(dAttnWeights, _lastAttnWeights);

            // Scale backward
            dScores = TensorOperations.Scale(dScores, _scale);

            // Gradient w.r.t. Q: dScores @ K
            Tensor dQ = TensorOperations.BatchedMatMul(dScores, _lastK);

            // Gradient w.r.t. K: dScores^T @ Q
            Tensor dScoresT = TransposeLast2D(dScores);
            Tensor dK = TensorOperations.BatchedMatMul(dScoresT, _lastQ);

            return (dQ, dK, dV);
        }

        private Tensor TransposeLast2D(Tensor t)
        {
            // For [batch, M, N] -> [batch, N, M]
            int batch = t.Shape[0];
            int M = t.Shape[1];
            int N = t.Shape[2];

            Tensor result = new Tensor(new[] { batch, N, M });

            for (int b = 0; b < batch; b++)
            {
                for (int i = 0; i < M; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        result[b, j, i] = t[b, i, j];
                    }
                }
            }

            return result;
        }

        private Tensor Softmax3D(Tensor t)
        {
            // Softmax along last dimension for 3D tensor
            int batch = t.Shape[0];
            int rows = t.Shape[1];
            int cols = t.Shape[2];

            Tensor result = new Tensor(t.Shape);

            for (int b = 0; b < batch; b++)
            {
                for (int i = 0; i < rows; i++)
                {
                    // Find max for numerical stability
                    float maxVal = float.NegativeInfinity;
                    for (int j = 0; j < cols; j++)
                    {
                        if (t[b, i, j] > maxVal) maxVal = t[b, i, j];
                    }

                    // Compute exp and sum
                    float sum = 0f;
                    for (int j = 0; j < cols; j++)
                    {
                        float expVal = (float)Math.Exp(t[b, i, j] - maxVal);
                        result[b, i, j] = expVal;
                        sum += expVal;
                    }

                    // Normalize
                    for (int j = 0; j < cols; j++)
                    {
                        result[b, i, j] /= sum;
                    }
                }
            }

            return result;
        }

        private Tensor SoftmaxBackward(Tensor dOut, Tensor softmaxOut)
        {
            // Gradient of softmax: y * (dy - sum(dy * y))
            int batch = dOut.Shape[0];
            int rows = dOut.Shape[1];
            int cols = dOut.Shape[2];

            Tensor result = new Tensor(dOut.Shape);

            for (int b = 0; b < batch; b++)
            {
                for (int i = 0; i < rows; i++)
                {
                    // Compute sum(dy * y)
                    float dotProduct = 0f;
                    for (int j = 0; j < cols; j++)
                    {
                        dotProduct += dOut[b, i, j] * softmaxOut[b, i, j];
                    }

                    // Compute gradient
                    for (int j = 0; j < cols; j++)
                    {
                        result[b, i, j] = softmaxOut[b, i, j] * (dOut[b, i, j] - dotProduct);
                    }
                }
            }

            return result;
        }

        private Tensor BroadcastAdd(Tensor a, Tensor b)
        {
            // a: [batch, seqLen, seqLen]
            // b: [1, seqLen, seqLen]
            // Result: [batch, seqLen, seqLen]
            int batch = a.Shape[0];
            int rows = a.Shape[1];
            int cols = a.Shape[2];

            Tensor result = new Tensor(a.Shape);

            for (int bIdx = 0; bIdx < batch; bIdx++)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        result[bIdx, i, j] = a[bIdx, i, j] + b[0, i, j];
                    }
                }
            }

            return result;
        }

        private Tensor ApplyDropout(Tensor t)
        {
            int batch = t.Shape[0];
            int rows = t.Shape[1];
            int cols = t.Shape[2];

            Tensor result = new Tensor(t.Shape);
            _lastDropoutMask = new Tensor(t.Shape);

            float scale = 1.0f / (1.0f - _dropout);

            for (int b = 0; b < batch; b++)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        if (_rng.NextDouble() >= _dropout)
                        {
                            _lastDropoutMask[b, i, j] = scale;
                            result[b, i, j] = t[b, i, j] * scale;
                        }
                        else
                        {
                            _lastDropoutMask[b, i, j] = 0f;
                            result[b, i, j] = 0f;
                        }
                    }
                }
            }

            return result;
        }
    }
}
