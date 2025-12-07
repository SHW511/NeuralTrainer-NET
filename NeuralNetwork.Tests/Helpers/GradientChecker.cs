namespace NeuralNetwork.Tests.Helpers
{
    /// <summary>
    /// Numerical gradient checking for verifying backpropagation correctness.
    /// Uses finite differences to approximate gradients and compare with analytical gradients.
    /// </summary>
    public static class GradientChecker
    {
        /// <summary>
        /// Check gradients for a layer using numerical differentiation.
        /// Returns the maximum relative error between analytical and numerical gradients.
        /// </summary>
        /// <param name="layer">The layer to check</param>
        /// <param name="input">Input data</param>
        /// <param name="epsilon">Perturbation size for finite differences</param>
        /// <returns>Maximum relative error (should be < 1e-5 for correct implementations)</returns>
        public static float CheckLayerGradients(Layer layer, float[,] input, float epsilon = 1e-4f)
        {
            int batchSize = input.GetLength(0);
            int inputDim = input.GetLength(1);

            // Build layer if not built
            if (!layer.Built)
            {
                layer.Build(new int[] { batchSize, inputDim });
            }

            // Forward pass
            var output = layer.Call(input);

            // Create gradient (all ones for simplicity)
            var gradOutput = CreateOnesLike(output);

            // Backward pass - get analytical input gradients
            var inputGradient = layer.Backward(gradOutput);

            // Check input gradients using finite differences
            if (inputGradient != null)
            {
                float maxError = CheckInputGradients(layer, input, output, inputGradient, epsilon);
                return maxError;
            }

            return 0f;
        }

        /// <summary>
        /// Check input gradients using finite differences.
        /// </summary>
        private static float CheckInputGradients(Layer layer, float[,] input, float[,] output, float[,] analyticalGrad, float epsilon)
        {
            float maxError = 0f;
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);

            // Only check first few elements for speed
            for (int i = 0; i < Math.Min(rows, 2); i++)
            {
                for (int j = 0; j < Math.Min(cols, Math.Min(analyticalGrad.GetLength(1), 3)); j++)
                {
                    // Make a copy of input
                    float[,] inputPerturbed = (float[,])input.Clone();

                    // f(x + epsilon)
                    inputPerturbed[i, j] = input[i, j] + epsilon;
                    var outputPlus = layer.Call(inputPerturbed);
                    float lossPlus = SumAll(outputPlus);

                    // f(x - epsilon)
                    inputPerturbed[i, j] = input[i, j] - epsilon;
                    var outputMinus = layer.Call(inputPerturbed);
                    float lossMinus = SumAll(outputMinus);

                    // Numerical gradient: (f(x+e) - f(x-e)) / 2e
                    float numericalGrad = (lossPlus - lossMinus) / (2 * epsilon);

                    // Get the analytical gradient (if it's valid for this index)
                    if (j < analyticalGrad.GetLength(1))
                    {
                        float analyticalGradValue = analyticalGrad[i, j];

                        // Relative error
                        float error = RelativeError(numericalGrad, analyticalGradValue);
                        if (error > maxError)
                            maxError = error;
                    }
                }
            }

            return maxError;
        }

        /// <summary>
        /// Check if gradients are numerically correct.
        /// </summary>
        /// <param name="layer">Layer to check</param>
        /// <param name="input">Test input</param>
        /// <param name="tolerance">Maximum allowed relative error</param>
        /// <returns>True if gradients are correct within tolerance</returns>
        public static bool AreGradientsCorrect(Layer layer, float[,] input, float tolerance = 1e-3f)
        {
            float maxError = CheckLayerGradients(layer, input);
            return maxError < tolerance;
        }

        /// <summary>
        /// Compute relative error between two values.
        /// </summary>
        public static float RelativeError(float a, float b)
        {
            float numerator = Math.Abs(a - b);
            float denominator = Math.Max(Math.Abs(a) + Math.Abs(b), 1e-8f);
            return numerator / denominator;
        }

        /// <summary>
        /// Sum all elements in a 2D array (used as simple loss function).
        /// </summary>
        private static float SumAll(float[,] array)
        {
            float sum = 0;
            for (int i = 0; i < array.GetLength(0); i++)
                for (int j = 0; j < array.GetLength(1); j++)
                    sum += array[i, j];
            return sum;
        }

        /// <summary>
        /// Create array of ones with same shape.
        /// </summary>
        private static float[,] CreateOnesLike(float[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            float[,] ones = new float[rows, cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    ones[i, j] = 1f;

            return ones;
        }

        /// <summary>
        /// Check numerical gradient for a function.
        /// </summary>
        /// <param name="f">Function to differentiate</param>
        /// <param name="x">Point at which to compute gradient</param>
        /// <param name="epsilon">Perturbation size</param>
        public static float NumericalGradient(Func<float, float> f, float x, float epsilon = 1e-4f)
        {
            return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon);
        }

        /// <summary>
        /// Check numerical gradient for a multi-variable function.
        /// </summary>
        public static float[] NumericalGradient(Func<float[], float> f, float[] x, float epsilon = 1e-4f)
        {
            float[] gradients = new float[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                float original = x[i];

                x[i] = original + epsilon;
                float fPlus = f(x);

                x[i] = original - epsilon;
                float fMinus = f(x);

                x[i] = original;

                gradients[i] = (fPlus - fMinus) / (2 * epsilon);
            }

            return gradients;
        }
    }
}
