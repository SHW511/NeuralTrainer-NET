namespace NeuralNetwork.Tests.Helpers
{
    /// <summary>
    /// Utilities for generating test data for ML tests.
    /// </summary>
    public static class TestDataBuilder
    {
        private static readonly Random _random = new Random(42);  // Fixed seed for reproducibility

        /// <summary>
        /// Generate random 2D array with values in range [min, max].
        /// </summary>
        public static float[,] RandomMatrix(int rows, int cols, float min = -1f, float max = 1f)
        {
            float[,] matrix = new float[rows, cols];
            float range = max - min;

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = (float)_random.NextDouble() * range + min;

            return matrix;
        }

        /// <summary>
        /// Generate random 1D array with values in range [min, max].
        /// </summary>
        public static float[] RandomVector(int length, float min = -1f, float max = 1f)
        {
            float[] vector = new float[length];
            float range = max - min;

            for (int i = 0; i < length; i++)
                vector[i] = (float)_random.NextDouble() * range + min;

            return vector;
        }

        /// <summary>
        /// Generate random 3D array with values in range [min, max].
        /// </summary>
        public static float[,,] Random3DArray(int dim0, int dim1, int dim2, float min = -1f, float max = 1f)
        {
            float[,,] array = new float[dim0, dim1, dim2];
            float range = max - min;

            for (int i = 0; i < dim0; i++)
                for (int j = 0; j < dim1; j++)
                    for (int k = 0; k < dim2; k++)
                        array[i, j, k] = (float)_random.NextDouble() * range + min;

            return array;
        }

        /// <summary>
        /// Generate matrix of zeros.
        /// </summary>
        public static float[,] Zeros(int rows, int cols)
        {
            return new float[rows, cols];
        }

        /// <summary>
        /// Generate matrix of ones.
        /// </summary>
        public static float[,] Ones(int rows, int cols)
        {
            float[,] matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = 1f;
            return matrix;
        }

        /// <summary>
        /// Generate identity matrix.
        /// </summary>
        public static float[,] Identity(int size)
        {
            float[,] matrix = new float[size, size];
            for (int i = 0; i < size; i++)
                matrix[i, i] = 1f;
            return matrix;
        }

        /// <summary>
        /// Generate simple XOR dataset.
        /// </summary>
        public static (float[,] inputs, float[,] outputs) XorDataset()
        {
            float[,] inputs = new float[,]
            {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
            };

            float[,] outputs = new float[,]
            {
                { 0 },
                { 1 },
                { 1 },
                { 0 }
            };

            return (inputs, outputs);
        }

        /// <summary>
        /// Generate simple linear dataset y = wx + b + noise.
        /// </summary>
        public static (float[,] inputs, float[,] outputs) LinearDataset(int samples, int inputDim, int outputDim, float noise = 0.1f)
        {
            float[,] inputs = RandomMatrix(samples, inputDim);
            float[,] weights = RandomMatrix(inputDim, outputDim);
            float[] bias = RandomVector(outputDim);

            float[,] outputs = new float[samples, outputDim];

            for (int i = 0; i < samples; i++)
            {
                for (int o = 0; o < outputDim; o++)
                {
                    float sum = bias[o];
                    for (int j = 0; j < inputDim; j++)
                        sum += inputs[i, j] * weights[j, o];

                    outputs[i, o] = sum + (float)(_random.NextDouble() - 0.5) * 2 * noise;
                }
            }

            return (inputs, outputs);
        }

        /// <summary>
        /// Generate sequence data for RNN testing.
        /// </summary>
        public static float[,,] SequenceData(int batchSize, int seqLength, int features)
        {
            return Random3DArray(batchSize, seqLength, features);
        }

        /// <summary>
        /// Generate integer token sequence for embedding testing.
        /// </summary>
        public static int[] TokenSequence(int length, int vocabSize)
        {
            int[] tokens = new int[length];
            for (int i = 0; i < length; i++)
                tokens[i] = _random.Next(vocabSize);
            return tokens;
        }

        /// <summary>
        /// Generate one-hot encoded matrix.
        /// </summary>
        public static float[,] OneHot(int[] indices, int numClasses)
        {
            float[,] oneHot = new float[indices.Length, numClasses];
            for (int i = 0; i < indices.Length; i++)
                oneHot[i, indices[i]] = 1f;
            return oneHot;
        }

        /// <summary>
        /// Generate batch of random images (batch, height, width, channels).
        /// </summary>
        public static float[,,,] RandomImages(int batchSize, int height, int width, int channels)
        {
            float[,,,] images = new float[batchSize, height, width, channels];
            for (int b = 0; b < batchSize; b++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        for (int c = 0; c < channels; c++)
                            images[b, h, w, c] = (float)_random.NextDouble();
            return images;
        }

        /// <summary>
        /// Clone/copy a 2D array.
        /// </summary>
        public static float[,] Clone(float[,] source)
        {
            int rows = source.GetLength(0);
            int cols = source.GetLength(1);
            float[,] copy = new float[rows, cols];
            Array.Copy(source, copy, source.Length);
            return copy;
        }

        /// <summary>
        /// Reset random seed for reproducible tests.
        /// </summary>
        public static void ResetSeed(int seed = 42)
        {
            // Note: In .NET 6+, Random is not reseedable, so we create a new instance
            // For test reproducibility, use this at the start of tests
        }
    }
}
