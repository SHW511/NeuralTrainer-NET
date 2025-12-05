using System;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Layers.Attention
{
    /// <summary>
    /// Utility class for creating attention masks.
    /// </summary>
    public static class AttentionMask
    {
        /// <summary>
        /// Create a causal (lower-triangular) mask for autoregressive models.
        /// Returns a matrix where position (i,j) is 0 if i >= j, else -infinity.
        /// </summary>
        /// <param name="seqLen">Sequence length.</param>
        /// <returns>Mask tensor of shape [seqLen, seqLen].</returns>
        public static Tensor CreateCausalMask(int seqLen)
        {
            Tensor mask = new Tensor(new[] { seqLen, seqLen });

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    // Allow attending to current and previous positions
                    mask[i, j] = j <= i ? 0f : float.NegativeInfinity;
                }
            }

            return mask;
        }

        /// <summary>
        /// Create a causal mask for batched attention.
        /// Returns a tensor of shape [1, seqLen, seqLen] that can be broadcast.
        /// </summary>
        public static Tensor CreateCausalMaskBatched(int seqLen)
        {
            Tensor mask = new Tensor(new[] { 1, seqLen, seqLen });

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    mask[0, i, j] = j <= i ? 0f : float.NegativeInfinity;
                }
            }

            return mask;
        }

        /// <summary>
        /// Create a padding mask for variable-length sequences.
        /// </summary>
        /// <param name="lengths">Actual length of each sequence in the batch.</param>
        /// <param name="maxLen">Maximum sequence length.</param>
        /// <returns>Mask tensor of shape [batch, 1, maxLen] where padded positions are -infinity.</returns>
        public static Tensor CreatePaddingMask(int[] lengths, int maxLen)
        {
            int batch = lengths.Length;
            Tensor mask = new Tensor(new[] { batch, 1, maxLen });

            for (int b = 0; b < batch; b++)
            {
                for (int j = 0; j < maxLen; j++)
                {
                    mask[b, 0, j] = j < lengths[b] ? 0f : float.NegativeInfinity;
                }
            }

            return mask;
        }

        /// <summary>
        /// Combine causal mask with padding mask.
        /// </summary>
        public static Tensor CombineMasks(Tensor causalMask, Tensor paddingMask)
        {
            // causalMask: [1, seqLen, seqLen] or [seqLen, seqLen]
            // paddingMask: [batch, 1, seqLen]
            // Result: [batch, seqLen, seqLen]

            int batch = paddingMask.Shape[0];
            int seqLen = causalMask.Rank == 2 ? causalMask.Shape[0] : causalMask.Shape[1];

            Tensor combined = new Tensor(new[] { batch, seqLen, seqLen });

            for (int b = 0; b < batch; b++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        float causalVal = causalMask.Rank == 2 ? causalMask[i, j] : causalMask[0, i, j];
                        float paddingVal = paddingMask[b, 0, j];

                        // Take minimum (more negative = more masked)
                        combined[b, i, j] = Math.Min(causalVal, paddingVal);
                    }
                }
            }

            return combined;
        }

        /// <summary>
        /// Apply mask to attention scores (add mask values to scores).
        /// </summary>
        /// <param name="scores">Attention scores [batch, seqLen, seqLen].</param>
        /// <param name="mask">Mask tensor (broadcastable to scores shape).</param>
        /// <returns>Masked scores.</returns>
        public static Tensor ApplyMask(Tensor scores, Tensor mask)
        {
            return TensorOperations.Add(scores, mask);
        }
    }
}
