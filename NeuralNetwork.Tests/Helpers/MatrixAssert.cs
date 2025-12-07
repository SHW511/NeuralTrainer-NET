using Xunit;

namespace NeuralNetwork.Tests.Helpers
{
    /// <summary>
    /// Assertion helpers for comparing floating-point arrays with tolerance.
    /// Essential for ML testing where exact equality is rarely achievable.
    /// </summary>
    public static class MatrixAssert
    {
        /// <summary>
        /// Assert two 1D arrays are approximately equal within tolerance.
        /// </summary>
        public static void AreAlmostEqual(float[] expected, float[] actual, float tolerance = 1e-5f, string? message = null)
        {
            Assert.Equal(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                var diff = Math.Abs(expected[i] - actual[i]);
                Assert.True(diff <= tolerance,
                    $"{message ?? "Arrays differ"} at index [{i}]: expected {expected[i]}, got {actual[i]}, diff {diff} > tolerance {tolerance}");
            }
        }

        /// <summary>
        /// Assert two 2D arrays are approximately equal within tolerance.
        /// </summary>
        public static void AreAlmostEqual(float[,] expected, float[,] actual, float tolerance = 1e-5f, string? message = null)
        {
            Assert.Equal(expected.GetLength(0), actual.GetLength(0));
            Assert.Equal(expected.GetLength(1), actual.GetLength(1));

            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    var diff = Math.Abs(expected[i, j] - actual[i, j]);
                    Assert.True(diff <= tolerance,
                        $"{message ?? "Matrices differ"} at [{i},{j}]: expected {expected[i, j]}, got {actual[i, j]}, diff {diff} > tolerance {tolerance}");
                }
            }
        }

        /// <summary>
        /// Assert two 3D arrays are approximately equal within tolerance.
        /// </summary>
        public static void AreAlmostEqual(float[,,] expected, float[,,] actual, float tolerance = 1e-5f, string? message = null)
        {
            Assert.Equal(expected.GetLength(0), actual.GetLength(0));
            Assert.Equal(expected.GetLength(1), actual.GetLength(1));
            Assert.Equal(expected.GetLength(2), actual.GetLength(2));

            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    for (int k = 0; k < expected.GetLength(2); k++)
                    {
                        var diff = Math.Abs(expected[i, j, k] - actual[i, j, k]);
                        Assert.True(diff <= tolerance,
                            $"{message ?? "Arrays differ"} at [{i},{j},{k}]: expected {expected[i, j, k]}, got {actual[i, j, k]}, diff {diff} > tolerance {tolerance}");
                    }
                }
            }
        }

        /// <summary>
        /// Check if array contains any NaN or Infinity values.
        /// </summary>
        public static void IsFinite(float[] array, string? message = null)
        {
            for (int i = 0; i < array.Length; i++)
            {
                Assert.True(float.IsFinite(array[i]),
                    $"{message ?? "Array contains non-finite value"} at [{i}]: {array[i]}");
            }
        }

        /// <summary>
        /// Check if 2D array contains any NaN or Infinity values.
        /// </summary>
        public static void IsFinite(float[,] array, string? message = null)
        {
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Assert.True(float.IsFinite(array[i, j]),
                        $"{message ?? "Matrix contains non-finite value"} at [{i},{j}]: {array[i, j]}");
                }
            }
        }

        /// <summary>
        /// Assert array has expected shape.
        /// </summary>
        public static void HasShape(float[,] array, int expectedRows, int expectedCols)
        {
            Assert.Equal(expectedRows, array.GetLength(0));
            Assert.Equal(expectedCols, array.GetLength(1));
        }

        /// <summary>
        /// Assert 1D array has expected length.
        /// </summary>
        public static void HasLength(float[] array, int expectedLength)
        {
            Assert.Equal(expectedLength, array.Length);
        }

        /// <summary>
        /// Compute mean absolute difference between two arrays.
        /// </summary>
        public static float MeanAbsoluteDifference(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Arrays must have same length");

            float sum = 0;
            for (int i = 0; i < a.Length; i++)
                sum += Math.Abs(a[i] - b[i]);

            return sum / a.Length;
        }

        /// <summary>
        /// Compute max absolute difference between two arrays.
        /// </summary>
        public static float MaxAbsoluteDifference(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Arrays must have same length");

            float max = 0;
            for (int i = 0; i < a.Length; i++)
            {
                float diff = Math.Abs(a[i] - b[i]);
                if (diff > max) max = diff;
            }

            return max;
        }

        /// <summary>
        /// Flatten a 2D array to 1D.
        /// </summary>
        public static float[] Flatten(float[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            float[] flat = new float[rows * cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    flat[i * cols + j] = array[i, j];

            return flat;
        }
    }
}
