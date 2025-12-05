using System;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Layers.Activations
{
    /// <summary>
    /// Gaussian Error Linear Unit (GELU) activation function.
    /// Used in Transformers (BERT, GPT, etc.) as the default activation in FFN.
    ///
    /// GELU(x) = x * Φ(x) where Φ is the CDF of the standard normal distribution.
    ///
    /// Fast approximation:
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    /// </summary>
    public static class GELU
    {
        private const float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/π)
        private const float COEFF = 0.044715f;

        /// <summary>
        /// Apply GELU activation to a 2D array.
        /// </summary>
        public static float[,] Apply(float[,] inputs)
        {
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            float[,] outputs = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    outputs[i, j] = GELUScalar(inputs[i, j]);
                }
            }

            return outputs;
        }

        /// <summary>
        /// Apply GELU activation to a Tensor.
        /// </summary>
        public static Tensor Apply(Tensor input)
        {
            Tensor output = new Tensor(input.Shape);

            for (int i = 0; i < input.Size; i++)
            {
                output.Data[i] = GELUScalar(input.Data[i]);
            }

            return output;
        }

        /// <summary>
        /// Compute GELU derivative for backward pass.
        /// </summary>
        public static float[,] Backward(float[,] inputs, float[,] gradOutput)
        {
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            float[,] gradInput = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    gradInput[i, j] = GELUDerivative(inputs[i, j]) * gradOutput[i, j];
                }
            }

            return gradInput;
        }

        /// <summary>
        /// Compute GELU derivative for Tensor backward pass.
        /// </summary>
        public static Tensor Backward(Tensor input, Tensor gradOutput)
        {
            Tensor gradInput = new Tensor(input.Shape);

            for (int i = 0; i < input.Size; i++)
            {
                gradInput.Data[i] = GELUDerivative(input.Data[i]) * gradOutput.Data[i];
            }

            return gradInput;
        }

        /// <summary>
        /// Scalar GELU using fast approximation.
        /// </summary>
        private static float GELUScalar(float x)
        {
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            return 0.5f * x * (1.0f + (float)Math.Tanh(inner));
        }

        /// <summary>
        /// Derivative of GELU using fast approximation.
        /// d/dx GELU(x) = 0.5 * (1 + tanh(z)) + 0.5 * x * sech²(z) * dz/dx
        /// where z = sqrt(2/π) * (x + 0.044715 * x³)
        /// and dz/dx = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
        /// </summary>
        private static float GELUDerivative(float x)
        {
            float x2 = x * x;
            float x3 = x2 * x;

            float z = SQRT_2_OVER_PI * (x + COEFF * x3);
            float tanhZ = (float)Math.Tanh(z);
            float sech2Z = 1.0f - tanhZ * tanhZ;  // sech²(z) = 1 - tanh²(z)

            float dzDx = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x2);

            return 0.5f * (1.0f + tanhZ) + 0.5f * x * sech2Z * dzDx;
        }

        /// <summary>
        /// Exact GELU using error function (slower but more accurate).
        /// </summary>
        public static float GELUExact(float x)
        {
            // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            return 0.5f * x * (1.0f + Erf(x / 1.4142135623730951f));
        }

        /// <summary>
        /// Error function approximation.
        /// </summary>
        private static float Erf(float x)
        {
            // Abramowitz and Stegun approximation
            float sign = x < 0 ? -1.0f : 1.0f;
            x = Math.Abs(x);

            const float a1 = 0.254829592f;
            const float a2 = -0.284496736f;
            const float a3 = 1.421413741f;
            const float a4 = -1.453152027f;
            const float a5 = 1.061405429f;
            const float p = 0.3275911f;

            float t = 1.0f / (1.0f + p * x);
            float y = 1.0f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (float)Math.Exp(-x * x);

            return sign * y;
        }
    }
}
