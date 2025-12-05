using System;
using System.Threading.Tasks;

namespace NeuralNetwork.Tensors
{
    /// <summary>
    /// CPU implementations of tensor operations with parallel processing.
    /// Uses Parallel.For for multi-threaded execution on available CPU cores.
    /// </summary>
    public static class TensorOperations
    {
        // Threshold for parallelization - small operations run faster single-threaded
        private const int PARALLEL_THRESHOLD = 64;

        /// <summary>
        /// Batched matrix multiplication: [B, M, K] x [B, K, N] -> [B, M, N]
        /// Also supports broadcasting when batch dimension is 1.
        /// Parallelized over batch and M dimensions.
        /// </summary>
        public static Tensor BatchedMatMul(Tensor a, Tensor b)
        {
            if (a.Rank != 3 || b.Rank != 3)
                throw new ArgumentException($"BatchedMatMul requires rank-3 tensors, got {a.Rank} and {b.Rank}");

            int batchA = a.Shape[0];
            int batchB = b.Shape[0];
            int M = a.Shape[1];
            int K = a.Shape[2];
            int N = b.Shape[2];

            if (a.Shape[2] != b.Shape[1])
                throw new ArgumentException($"Inner dimensions must match: {a.Shape[2]} vs {b.Shape[1]}");

            // Handle broadcasting
            int batch = Math.Max(batchA, batchB);
            if (batchA != batchB && batchA != 1 && batchB != 1)
                throw new ArgumentException($"Batch dimensions not broadcastable: {batchA} vs {batchB}");

            Tensor result = new Tensor(new[] { batch, M, N });
            float[] resultData = result.Data;
            float[] aData = a.Data;
            float[] bData = b.Data;

            // Parallelize over batch * M (outer two loops combined)
            int totalRows = batch * M;

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, totalRows, row =>
                {
                    int bIdx = row / M;
                    int i = row % M;

                    int aIdx = batchA == 1 ? 0 : bIdx;
                    int bIdx2 = batchB == 1 ? 0 : bIdx;

                    int aRowOffset = aIdx * M * K + i * K;
                    int resultRowOffset = bIdx * M * N + i * N;

                    for (int j = 0; j < N; j++)
                    {
                        float sum = 0f;
                        int bColOffset = bIdx2 * K * N + j;

                        for (int k = 0; k < K; k++)
                        {
                            sum += aData[aRowOffset + k] * bData[bColOffset + k * N];
                        }
                        resultData[resultRowOffset + j] = sum;
                    }
                });
            }
            else
            {
                // Sequential for small operations
                for (int bIdx = 0; bIdx < batch; bIdx++)
                {
                    int aIdx = batchA == 1 ? 0 : bIdx;
                    int bIdx2 = batchB == 1 ? 0 : bIdx;

                    for (int i = 0; i < M; i++)
                    {
                        int aRowOffset = aIdx * M * K + i * K;
                        int resultRowOffset = bIdx * M * N + i * N;

                        for (int j = 0; j < N; j++)
                        {
                            float sum = 0f;
                            int bColOffset = bIdx2 * K * N + j;

                            for (int k = 0; k < K; k++)
                            {
                                sum += aData[aRowOffset + k] * bData[bColOffset + k * N];
                            }
                            resultData[resultRowOffset + j] = sum;
                        }
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Standard 2D matrix multiplication: [M, K] x [K, N] -> [M, N]
        /// Parallelized over M dimension.
        /// </summary>
        public static Tensor MatMul(Tensor a, Tensor b)
        {
            if (a.Rank != 2 || b.Rank != 2)
                throw new ArgumentException($"MatMul requires rank-2 tensors, got {a.Rank} and {b.Rank}");

            int M = a.Shape[0];
            int K = a.Shape[1];
            int N = b.Shape[1];

            if (a.Shape[1] != b.Shape[0])
                throw new ArgumentException($"Inner dimensions must match: {a.Shape[1]} vs {b.Shape[0]}");

            Tensor result = new Tensor(new[] { M, N });
            float[] resultData = result.Data;
            float[] aData = a.Data;
            float[] bData = b.Data;

            if (M >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, M, i =>
                {
                    int aRowOffset = i * K;
                    int resultRowOffset = i * N;

                    for (int j = 0; j < N; j++)
                    {
                        float sum = 0f;
                        for (int k = 0; k < K; k++)
                        {
                            sum += aData[aRowOffset + k] * bData[k * N + j];
                        }
                        resultData[resultRowOffset + j] = sum;
                    }
                });
            }
            else
            {
                for (int i = 0; i < M; i++)
                {
                    int aRowOffset = i * K;
                    int resultRowOffset = i * N;

                    for (int j = 0; j < N; j++)
                    {
                        float sum = 0f;
                        for (int k = 0; k < K; k++)
                        {
                            sum += aData[aRowOffset + k] * bData[k * N + j];
                        }
                        resultData[resultRowOffset + j] = sum;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Element-wise addition with broadcasting.
        /// </summary>
        public static Tensor Add(Tensor a, Tensor b)
        {
            // Simple case: same shape
            if (ShapesEqual(a.Shape, b.Shape))
            {
                Tensor result = new Tensor(a.Shape);
                float[] resultData = result.Data;
                float[] aData = a.Data;
                float[] bData = b.Data;
                int size = a.Size;

                if (size >= PARALLEL_THRESHOLD * 16)
                {
                    Parallel.For(0, size, i =>
                    {
                        resultData[i] = aData[i] + bData[i];
                    });
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        resultData[i] = aData[i] + bData[i];
                    }
                }
                return result;
            }

            // Broadcasting case
            int[] resultShape = BroadcastShapes(a.Shape, b.Shape);
            Tensor output = new Tensor(resultShape);
            BroadcastOperation(a, b, output, (x, y) => x + y);
            return output;
        }

        /// <summary>
        /// Element-wise subtraction with broadcasting.
        /// </summary>
        public static Tensor Subtract(Tensor a, Tensor b)
        {
            if (ShapesEqual(a.Shape, b.Shape))
            {
                Tensor result = new Tensor(a.Shape);
                float[] resultData = result.Data;
                float[] aData = a.Data;
                float[] bData = b.Data;
                int size = a.Size;

                if (size >= PARALLEL_THRESHOLD * 16)
                {
                    Parallel.For(0, size, i =>
                    {
                        resultData[i] = aData[i] - bData[i];
                    });
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        resultData[i] = aData[i] - bData[i];
                    }
                }
                return result;
            }

            int[] resultShape = BroadcastShapes(a.Shape, b.Shape);
            Tensor output = new Tensor(resultShape);
            BroadcastOperation(a, b, output, (x, y) => x - y);
            return output;
        }

        /// <summary>
        /// Element-wise multiplication with broadcasting.
        /// </summary>
        public static Tensor Multiply(Tensor a, Tensor b)
        {
            if (ShapesEqual(a.Shape, b.Shape))
            {
                Tensor result = new Tensor(a.Shape);
                float[] resultData = result.Data;
                float[] aData = a.Data;
                float[] bData = b.Data;
                int size = a.Size;

                if (size >= PARALLEL_THRESHOLD * 16)
                {
                    Parallel.For(0, size, i =>
                    {
                        resultData[i] = aData[i] * bData[i];
                    });
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        resultData[i] = aData[i] * bData[i];
                    }
                }
                return result;
            }

            int[] resultShape = BroadcastShapes(a.Shape, b.Shape);
            Tensor output = new Tensor(resultShape);
            BroadcastOperation(a, b, output, (x, y) => x * y);
            return output;
        }

        /// <summary>
        /// Element-wise division with broadcasting.
        /// </summary>
        public static Tensor Divide(Tensor a, Tensor b)
        {
            if (ShapesEqual(a.Shape, b.Shape))
            {
                Tensor result = new Tensor(a.Shape);
                float[] resultData = result.Data;
                float[] aData = a.Data;
                float[] bData = b.Data;
                int size = a.Size;

                if (size >= PARALLEL_THRESHOLD * 16)
                {
                    Parallel.For(0, size, i =>
                    {
                        resultData[i] = aData[i] / bData[i];
                    });
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        resultData[i] = aData[i] / bData[i];
                    }
                }
                return result;
            }

            int[] resultShape = BroadcastShapes(a.Shape, b.Shape);
            Tensor output = new Tensor(resultShape);
            BroadcastOperation(a, b, output, (x, y) => x / y);
            return output;
        }

        /// <summary>
        /// Scale tensor by a scalar value.
        /// </summary>
        public static Tensor Scale(Tensor a, float scalar)
        {
            Tensor result = new Tensor(a.Shape);
            float[] resultData = result.Data;
            float[] aData = a.Data;
            int size = a.Size;

            if (size >= PARALLEL_THRESHOLD * 16)
            {
                Parallel.For(0, size, i =>
                {
                    resultData[i] = aData[i] * scalar;
                });
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    resultData[i] = aData[i] * scalar;
                }
            }
            return result;
        }

        /// <summary>
        /// Add a scalar to all elements.
        /// </summary>
        public static Tensor AddScalar(Tensor a, float scalar)
        {
            Tensor result = new Tensor(a.Shape);
            float[] resultData = result.Data;
            float[] aData = a.Data;
            int size = a.Size;

            if (size >= PARALLEL_THRESHOLD * 16)
            {
                Parallel.For(0, size, i =>
                {
                    resultData[i] = aData[i] + scalar;
                });
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    resultData[i] = aData[i] + scalar;
                }
            }
            return result;
        }

        /// <summary>
        /// Element-wise square root.
        /// </summary>
        public static Tensor Sqrt(Tensor a)
        {
            Tensor result = new Tensor(a.Shape);
            float[] resultData = result.Data;
            float[] aData = a.Data;
            int size = a.Size;

            if (size >= PARALLEL_THRESHOLD * 16)
            {
                Parallel.For(0, size, i =>
                {
                    resultData[i] = MathF.Sqrt(aData[i]);
                });
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    resultData[i] = MathF.Sqrt(aData[i]);
                }
            }
            return result;
        }

        /// <summary>
        /// Element-wise power.
        /// </summary>
        public static Tensor Pow(Tensor a, float power)
        {
            Tensor result = new Tensor(a.Shape);
            float[] resultData = result.Data;
            float[] aData = a.Data;
            int size = a.Size;

            if (size >= PARALLEL_THRESHOLD * 16)
            {
                Parallel.For(0, size, i =>
                {
                    resultData[i] = MathF.Pow(aData[i], power);
                });
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    resultData[i] = MathF.Pow(aData[i], power);
                }
            }
            return result;
        }

        /// <summary>
        /// Element-wise exponential.
        /// </summary>
        public static Tensor Exp(Tensor a)
        {
            Tensor result = new Tensor(a.Shape);
            float[] resultData = result.Data;
            float[] aData = a.Data;
            int size = a.Size;

            if (size >= PARALLEL_THRESHOLD * 16)
            {
                Parallel.For(0, size, i =>
                {
                    resultData[i] = MathF.Exp(aData[i]);
                });
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    resultData[i] = MathF.Exp(aData[i]);
                }
            }
            return result;
        }

        /// <summary>
        /// Element-wise natural logarithm.
        /// </summary>
        public static Tensor Log(Tensor a)
        {
            Tensor result = new Tensor(a.Shape);
            float[] resultData = result.Data;
            float[] aData = a.Data;
            int size = a.Size;

            if (size >= PARALLEL_THRESHOLD * 16)
            {
                Parallel.For(0, size, i =>
                {
                    resultData[i] = MathF.Log(aData[i]);
                });
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    resultData[i] = MathF.Log(aData[i]);
                }
            }
            return result;
        }

        /// <summary>
        /// Element-wise tanh.
        /// </summary>
        public static Tensor Tanh(Tensor a)
        {
            Tensor result = new Tensor(a.Shape);
            float[] resultData = result.Data;
            float[] aData = a.Data;
            int size = a.Size;

            if (size >= PARALLEL_THRESHOLD * 16)
            {
                Parallel.For(0, size, i =>
                {
                    resultData[i] = MathF.Tanh(aData[i]);
                });
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    resultData[i] = MathF.Tanh(aData[i]);
                }
            }
            return result;
        }

        /// <summary>
        /// Sum all elements.
        /// </summary>
        public static float Sum(Tensor a)
        {
            float[] aData = a.Data;
            int size = a.Size;

            if (size >= PARALLEL_THRESHOLD * 32)
            {
                // Parallel reduction
                int numThreads = Environment.ProcessorCount;
                float[] partialSums = new float[numThreads];
                int chunkSize = (size + numThreads - 1) / numThreads;

                Parallel.For(0, numThreads, threadIdx =>
                {
                    int start = threadIdx * chunkSize;
                    int end = Math.Min(start + chunkSize, size);
                    float localSum = 0f;
                    for (int i = start; i < end; i++)
                    {
                        localSum += aData[i];
                    }
                    partialSums[threadIdx] = localSum;
                });

                float sum = 0f;
                for (int i = 0; i < numThreads; i++)
                    sum += partialSums[i];
                return sum;
            }
            else
            {
                float sum = 0f;
                for (int i = 0; i < size; i++)
                {
                    sum += aData[i];
                }
                return sum;
            }
        }

        /// <summary>
        /// Sum along a specific axis.
        /// </summary>
        public static Tensor SumAxis(Tensor a, int axis, bool keepDims = false)
        {
            if (axis < 0) axis += a.Rank;
            if (axis < 0 || axis >= a.Rank)
                throw new ArgumentException($"Invalid axis {axis} for rank {a.Rank}");

            int[] newShape;
            if (keepDims)
            {
                newShape = (int[])a.Shape.Clone();
                newShape[axis] = 1;
            }
            else
            {
                newShape = new int[a.Rank - 1];
                for (int i = 0, j = 0; i < a.Rank; i++)
                {
                    if (i != axis) newShape[j++] = a.Shape[i];
                }
            }

            Tensor result = new Tensor(newShape);
            float[] resultData = result.Data;

            // Compute strides for faster indexing
            int[] aStrides = ComputeStrides(a.Shape);
            int[] resultStrides = ComputeStrides(newShape);
            int axisSize = a.Shape[axis];
            int axisStride = aStrides[axis];

            if (result.Size >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, result.Size, i =>
                {
                    // Convert result flat index to multi-dim
                    int[] resultIndices = FlatToMulti(i, newShape, resultStrides);

                    // Build source base index
                    int[] sourceIndices = new int[a.Rank];
                    int srcIdx = 0;
                    for (int d = 0; d < a.Rank; d++)
                    {
                        if (d == axis)
                            sourceIndices[d] = 0;
                        else
                        {
                            if (keepDims)
                                sourceIndices[d] = resultIndices[d];
                            else
                                sourceIndices[d] = resultIndices[srcIdx++];
                        }
                    }

                    // Compute base flat index in source
                    int baseIdx = MultiToFlat(sourceIndices, aStrides);

                    // Sum over axis
                    float sum = 0f;
                    for (int k = 0; k < axisSize; k++)
                    {
                        sum += a.Data[baseIdx + k * axisStride];
                    }
                    resultData[i] = sum;
                });
            }
            else
            {
                for (int i = 0; i < result.Size; i++)
                {
                    int[] resultIndices = FlatToMulti(i, newShape, resultStrides);

                    int[] sourceIndices = new int[a.Rank];
                    int srcIdx = 0;
                    for (int d = 0; d < a.Rank; d++)
                    {
                        if (d == axis)
                            sourceIndices[d] = 0;
                        else
                        {
                            if (keepDims)
                                sourceIndices[d] = resultIndices[d];
                            else
                                sourceIndices[d] = resultIndices[srcIdx++];
                        }
                    }

                    int baseIdx = MultiToFlat(sourceIndices, aStrides);

                    float sum = 0f;
                    for (int k = 0; k < axisSize; k++)
                    {
                        sum += a.Data[baseIdx + k * axisStride];
                    }
                    resultData[i] = sum;
                }
            }

            return result;
        }

        /// <summary>
        /// Mean along a specific axis.
        /// </summary>
        public static Tensor MeanAxis(Tensor a, int axis, bool keepDims = false)
        {
            Tensor summed = SumAxis(a, axis, keepDims);
            return Scale(summed, 1.0f / a.Shape[axis < 0 ? axis + a.Rank : axis]);
        }

        /// <summary>
        /// Variance along a specific axis.
        /// </summary>
        public static Tensor VarAxis(Tensor a, int axis, bool keepDims = false)
        {
            if (axis < 0) axis += a.Rank;

            Tensor mean = MeanAxis(a, axis, keepDims: true);
            Tensor diff = Subtract(a, mean);
            Tensor squared = Multiply(diff, diff);
            return MeanAxis(squared, axis, keepDims);
        }

        /// <summary>
        /// Maximum element.
        /// </summary>
        public static float Max(Tensor a)
        {
            float[] aData = a.Data;
            int size = a.Size;

            if (size >= PARALLEL_THRESHOLD * 32)
            {
                int numThreads = Environment.ProcessorCount;
                float[] partialMax = new float[numThreads];
                for (int i = 0; i < numThreads; i++)
                    partialMax[i] = float.NegativeInfinity;

                int chunkSize = (size + numThreads - 1) / numThreads;

                Parallel.For(0, numThreads, threadIdx =>
                {
                    int start = threadIdx * chunkSize;
                    int end = Math.Min(start + chunkSize, size);
                    float localMax = float.NegativeInfinity;
                    for (int i = start; i < end; i++)
                    {
                        if (aData[i] > localMax) localMax = aData[i];
                    }
                    partialMax[threadIdx] = localMax;
                });

                float max = float.NegativeInfinity;
                for (int i = 0; i < numThreads; i++)
                    if (partialMax[i] > max) max = partialMax[i];
                return max;
            }
            else
            {
                float max = float.NegativeInfinity;
                for (int i = 0; i < size; i++)
                {
                    if (aData[i] > max) max = aData[i];
                }
                return max;
            }
        }

        /// <summary>
        /// Maximum along an axis.
        /// </summary>
        public static Tensor MaxAxis(Tensor a, int axis, bool keepDims = false)
        {
            if (axis < 0) axis += a.Rank;
            if (axis < 0 || axis >= a.Rank)
                throw new ArgumentException($"Invalid axis {axis} for rank {a.Rank}");

            int[] newShape;
            if (keepDims)
            {
                newShape = (int[])a.Shape.Clone();
                newShape[axis] = 1;
            }
            else
            {
                newShape = new int[a.Rank - 1];
                for (int i = 0, j = 0; i < a.Rank; i++)
                {
                    if (i != axis) newShape[j++] = a.Shape[i];
                }
            }

            Tensor result = new Tensor(newShape);
            result.Fill(float.NegativeInfinity);
            float[] resultData = result.Data;

            int[] aStrides = ComputeStrides(a.Shape);
            int[] resultStrides = ComputeStrides(newShape);
            int axisSize = a.Shape[axis];
            int axisStride = aStrides[axis];

            if (result.Size >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, result.Size, i =>
                {
                    int[] resultIndices = FlatToMulti(i, newShape, resultStrides);

                    int[] sourceIndices = new int[a.Rank];
                    int srcIdx = 0;
                    for (int d = 0; d < a.Rank; d++)
                    {
                        if (d == axis)
                            sourceIndices[d] = 0;
                        else
                        {
                            if (keepDims)
                                sourceIndices[d] = resultIndices[d];
                            else
                                sourceIndices[d] = resultIndices[srcIdx++];
                        }
                    }

                    int baseIdx = MultiToFlat(sourceIndices, aStrides);

                    float maxVal = float.NegativeInfinity;
                    for (int k = 0; k < axisSize; k++)
                    {
                        float val = a.Data[baseIdx + k * axisStride];
                        if (val > maxVal) maxVal = val;
                    }
                    resultData[i] = maxVal;
                });
            }
            else
            {
                for (int i = 0; i < result.Size; i++)
                {
                    int[] resultIndices = FlatToMulti(i, newShape, resultStrides);

                    int[] sourceIndices = new int[a.Rank];
                    int srcIdx = 0;
                    for (int d = 0; d < a.Rank; d++)
                    {
                        if (d == axis)
                            sourceIndices[d] = 0;
                        else
                        {
                            if (keepDims)
                                sourceIndices[d] = resultIndices[d];
                            else
                                sourceIndices[d] = resultIndices[srcIdx++];
                        }
                    }

                    int baseIdx = MultiToFlat(sourceIndices, aStrides);

                    float maxVal = float.NegativeInfinity;
                    for (int k = 0; k < axisSize; k++)
                    {
                        float val = a.Data[baseIdx + k * axisStride];
                        if (val > maxVal) maxVal = val;
                    }
                    resultData[i] = maxVal;
                }
            }

            return result;
        }

        /// <summary>
        /// Softmax along the last axis. Optimized and parallelized.
        /// </summary>
        public static Tensor Softmax(Tensor a, int axis = -1)
        {
            if (axis < 0) axis += a.Rank;

            // For 3D tensors with axis=-1 (most common in attention), use optimized path
            if (a.Rank == 3 && axis == 2)
            {
                return Softmax3DLastAxis(a);
            }

            // General implementation
            Tensor maxVals = MaxAxis(a, axis, keepDims: true);
            Tensor shifted = Subtract(a, maxVals);
            Tensor exps = Exp(shifted);
            Tensor sumExps = SumAxis(exps, axis, keepDims: true);
            return Divide(exps, sumExps);
        }

        /// <summary>
        /// Optimized softmax for 3D tensors along the last axis.
        /// Common in attention computations.
        /// </summary>
        private static Tensor Softmax3DLastAxis(Tensor a)
        {
            int batch = a.Shape[0];
            int seqLen = a.Shape[1];
            int dim = a.Shape[2];

            Tensor result = new Tensor(a.Shape);
            float[] aData = a.Data;
            float[] resultData = result.Data;

            int totalRows = batch * seqLen;

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, totalRows, row =>
                {
                    int offset = row * dim;

                    // Find max for numerical stability
                    float maxVal = float.NegativeInfinity;
                    for (int d = 0; d < dim; d++)
                    {
                        if (aData[offset + d] > maxVal)
                            maxVal = aData[offset + d];
                    }

                    // Compute exp and sum
                    float sumExp = 0f;
                    for (int d = 0; d < dim; d++)
                    {
                        float exp = MathF.Exp(aData[offset + d] - maxVal);
                        resultData[offset + d] = exp;
                        sumExp += exp;
                    }

                    // Normalize
                    float invSum = 1f / sumExp;
                    for (int d = 0; d < dim; d++)
                    {
                        resultData[offset + d] *= invSum;
                    }
                });
            }
            else
            {
                for (int row = 0; row < totalRows; row++)
                {
                    int offset = row * dim;

                    float maxVal = float.NegativeInfinity;
                    for (int d = 0; d < dim; d++)
                    {
                        if (aData[offset + d] > maxVal)
                            maxVal = aData[offset + d];
                    }

                    float sumExp = 0f;
                    for (int d = 0; d < dim; d++)
                    {
                        float exp = MathF.Exp(aData[offset + d] - maxVal);
                        resultData[offset + d] = exp;
                        sumExp += exp;
                    }

                    float invSum = 1f / sumExp;
                    for (int d = 0; d < dim; d++)
                    {
                        resultData[offset + d] *= invSum;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Concatenate tensors along an axis.
        /// </summary>
        public static Tensor Concat(Tensor[] tensors, int axis)
        {
            if (tensors == null || tensors.Length == 0)
                throw new ArgumentException("Must provide at least one tensor");

            if (axis < 0) axis += tensors[0].Rank;

            // Validate shapes
            int totalAxisSize = 0;
            for (int i = 0; i < tensors.Length; i++)
            {
                if (tensors[i].Rank != tensors[0].Rank)
                    throw new ArgumentException("All tensors must have the same rank");

                for (int d = 0; d < tensors[0].Rank; d++)
                {
                    if (d != axis && tensors[i].Shape[d] != tensors[0].Shape[d])
                        throw new ArgumentException($"Shapes must match except on concat axis");
                }
                totalAxisSize += tensors[i].Shape[axis];
            }

            int[] newShape = (int[])tensors[0].Shape.Clone();
            newShape[axis] = totalAxisSize;

            Tensor result = new Tensor(newShape);
            int[] resultStrides = ComputeStrides(newShape);

            // Copy data
            int axisOffset = 0;

            foreach (var tensor in tensors)
            {
                int[] tensorStrides = ComputeStrides(tensor.Shape);
                int tensorSize = tensor.Size;
                int tensorAxisSize = tensor.Shape[axis];
                int localAxisOffset = axisOffset;

                if (tensorSize >= PARALLEL_THRESHOLD)
                {
                    Parallel.For(0, tensorSize, i =>
                    {
                        int[] srcIndices = FlatToMulti(i, tensor.Shape, tensorStrides);
                        int[] dstIndices = (int[])srcIndices.Clone();
                        dstIndices[axis] += localAxisOffset;
                        int dstFlat = MultiToFlat(dstIndices, resultStrides);
                        result.Data[dstFlat] = tensor.Data[i];
                    });
                }
                else
                {
                    for (int i = 0; i < tensorSize; i++)
                    {
                        int[] srcIndices = FlatToMulti(i, tensor.Shape, tensorStrides);
                        int[] dstIndices = (int[])srcIndices.Clone();
                        dstIndices[axis] += localAxisOffset;
                        int dstFlat = MultiToFlat(dstIndices, resultStrides);
                        result.Data[dstFlat] = tensor.Data[i];
                    }
                }

                axisOffset += tensorAxisSize;
            }

            return result;
        }

        /// <summary>
        /// Split tensor into N equal parts along an axis.
        /// </summary>
        public static Tensor[] Split(Tensor a, int numSplits, int axis)
        {
            if (axis < 0) axis += a.Rank;

            if (a.Shape[axis] % numSplits != 0)
                throw new ArgumentException($"Cannot split dimension {a.Shape[axis]} into {numSplits} equal parts");

            int splitSize = a.Shape[axis] / numSplits;
            Tensor[] results = new Tensor[numSplits];

            int[] newShape = (int[])a.Shape.Clone();
            newShape[axis] = splitSize;

            int[] aStrides = ComputeStrides(a.Shape);
            int[] resultStrides = ComputeStrides(newShape);

            for (int s = 0; s < numSplits; s++)
            {
                results[s] = new Tensor(newShape);
                int splitOffset = s * splitSize;
                int resultSize = results[s].Size;

                if (resultSize >= PARALLEL_THRESHOLD)
                {
                    int localSplitOffset = splitOffset;
                    Tensor localResult = results[s];

                    Parallel.For(0, resultSize, i =>
                    {
                        int[] dstIndices = FlatToMulti(i, newShape, resultStrides);
                        int[] srcIndices = (int[])dstIndices.Clone();
                        srcIndices[axis] += localSplitOffset;
                        int srcFlat = MultiToFlat(srcIndices, aStrides);
                        localResult.Data[i] = a.Data[srcFlat];
                    });
                }
                else
                {
                    for (int i = 0; i < resultSize; i++)
                    {
                        int[] dstIndices = FlatToMulti(i, newShape, resultStrides);
                        int[] srcIndices = (int[])dstIndices.Clone();
                        srcIndices[axis] += splitOffset;
                        int srcFlat = MultiToFlat(srcIndices, aStrides);
                        results[s].Data[i] = a.Data[srcFlat];
                    }
                }
            }

            return results;
        }

        #region Helper Methods

        private static bool ShapesEqual(int[] a, int[] b)
        {
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i]) return false;
            }
            return true;
        }

        private static int[] BroadcastShapes(int[] a, int[] b)
        {
            int maxRank = Math.Max(a.Length, b.Length);
            int[] result = new int[maxRank];

            for (int i = 0; i < maxRank; i++)
            {
                int dimA = i < a.Length ? a[a.Length - 1 - i] : 1;
                int dimB = i < b.Length ? b[b.Length - 1 - i] : 1;

                if (dimA != dimB && dimA != 1 && dimB != 1)
                    throw new ArgumentException($"Shapes not broadcastable: dims {dimA} vs {dimB}");

                result[maxRank - 1 - i] = Math.Max(dimA, dimB);
            }

            return result;
        }

        private static void BroadcastOperation(Tensor a, Tensor b, Tensor output, Func<float, float, float> op)
        {
            int[] aStrides = ComputeStrides(a.Shape);
            int[] bStrides = ComputeStrides(b.Shape);
            int[] outStrides = ComputeStrides(output.Shape);

            if (output.Size >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, output.Size, i =>
                {
                    int[] outIndices = FlatToMulti(i, output.Shape, outStrides);

                    int aFlat = 0;
                    for (int d = 0; d < a.Rank; d++)
                    {
                        int outIdx = d + (output.Rank - a.Rank);
                        int idx = a.Shape[d] == 1 ? 0 : outIndices[outIdx];
                        aFlat += idx * aStrides[d];
                    }

                    int bFlat = 0;
                    for (int d = 0; d < b.Rank; d++)
                    {
                        int outIdx = d + (output.Rank - b.Rank);
                        int idx = b.Shape[d] == 1 ? 0 : outIndices[outIdx];
                        bFlat += idx * bStrides[d];
                    }

                    output.Data[i] = op(a.Data[aFlat], b.Data[bFlat]);
                });
            }
            else
            {
                int[] outIndices = new int[output.Rank];

                for (int i = 0; i < output.Size; i++)
                {
                    int remaining = i;
                    for (int d = output.Rank - 1; d >= 0; d--)
                    {
                        outIndices[d] = remaining % output.Shape[d];
                        remaining /= output.Shape[d];
                    }

                    int aFlat = 0;
                    for (int d = 0; d < a.Rank; d++)
                    {
                        int outIdx = d + (output.Rank - a.Rank);
                        int idx = a.Shape[d] == 1 ? 0 : outIndices[outIdx];
                        aFlat += idx * aStrides[d];
                    }

                    int bFlat = 0;
                    for (int d = 0; d < b.Rank; d++)
                    {
                        int outIdx = d + (output.Rank - b.Rank);
                        int idx = b.Shape[d] == 1 ? 0 : outIndices[outIdx];
                        bFlat += idx * bStrides[d];
                    }

                    output.Data[i] = op(a.Data[aFlat], b.Data[bFlat]);
                }
            }
        }

        private static int[] ComputeStrides(int[] shape)
        {
            int[] strides = new int[shape.Length];
            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }

        private static int[] FlatToMulti(int flatIndex, int[] shape, int[] strides)
        {
            int[] indices = new int[shape.Length];
            for (int d = 0; d < shape.Length; d++)
            {
                indices[d] = flatIndex / strides[d];
                flatIndex %= strides[d];
            }
            return indices;
        }

        private static int MultiToFlat(int[] indices, int[] strides)
        {
            int flat = 0;
            for (int d = 0; d < indices.Length; d++)
            {
                flat += indices[d] * strides[d];
            }
            return flat;
        }

        #endregion
    }
}
