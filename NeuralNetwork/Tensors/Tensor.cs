using System;
using System.Text;

namespace NeuralNetwork.Tensors
{
    /// <summary>
    /// Multi-dimensional tensor class with proper shape management.
    /// Supports arbitrary dimensions with row-major memory layout.
    /// </summary>
    public class Tensor
    {
        private readonly float[] _data;
        private readonly int[] _shape;
        private readonly int[] _strides;

        /// <summary>
        /// Raw data array in row-major order.
        /// </summary>
        public float[] Data => _data;

        /// <summary>
        /// Shape of the tensor (dimensions).
        /// </summary>
        public int[] Shape => (int[])_shape.Clone();

        /// <summary>
        /// Number of dimensions (rank).
        /// </summary>
        public int Rank => _shape.Length;

        /// <summary>
        /// Total number of elements.
        /// </summary>
        public int Size => _data.Length;

        /// <summary>
        /// Strides for each dimension (elements to skip).
        /// </summary>
        public int[] Strides => (int[])_strides.Clone();

        /// <summary>
        /// Create a zero-initialized tensor with the given shape.
        /// </summary>
        public Tensor(int[] shape)
        {
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("Shape cannot be null or empty");

            _shape = (int[])shape.Clone();
            _strides = ComputeStrides(_shape);
            _data = new float[ComputeSize(_shape)];
        }

        /// <summary>
        /// Create a tensor from existing data with the given shape.
        /// </summary>
        public Tensor(float[] data, int[] shape)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("Shape cannot be null or empty");

            int expectedSize = ComputeSize(shape);
            if (data.Length != expectedSize)
                throw new ArgumentException($"Data length {data.Length} doesn't match shape size {expectedSize}");

            _shape = (int[])shape.Clone();
            _strides = ComputeStrides(_shape);
            _data = (float[])data.Clone();
        }

        /// <summary>
        /// Create a tensor from a 2D array.
        /// </summary>
        public static Tensor FromArray(float[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            float[] data = new float[rows * cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    data[i * cols + j] = array[i, j];
                }
            }

            return new Tensor(data, new[] { rows, cols });
        }

        /// <summary>
        /// Create a tensor from a 3D array.
        /// </summary>
        public static Tensor FromArray(float[,,] array)
        {
            int d0 = array.GetLength(0);
            int d1 = array.GetLength(1);
            int d2 = array.GetLength(2);
            float[] data = new float[d0 * d1 * d2];

            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        data[i * d1 * d2 + j * d2 + k] = array[i, j, k];
                    }
                }
            }

            return new Tensor(data, new[] { d0, d1, d2 });
        }

        /// <summary>
        /// Convert tensor to 2D array (must be rank 2).
        /// </summary>
        public float[,] ToArray2D()
        {
            if (Rank != 2)
                throw new InvalidOperationException($"Cannot convert rank {Rank} tensor to 2D array");

            float[,] result = new float[_shape[0], _shape[1]];
            for (int i = 0; i < _shape[0]; i++)
            {
                for (int j = 0; j < _shape[1]; j++)
                {
                    result[i, j] = _data[i * _shape[1] + j];
                }
            }
            return result;
        }

        /// <summary>
        /// Convert tensor to 3D array (must be rank 3).
        /// </summary>
        public float[,,] ToArray3D()
        {
            if (Rank != 3)
                throw new InvalidOperationException($"Cannot convert rank {Rank} tensor to 3D array");

            float[,,] result = new float[_shape[0], _shape[1], _shape[2]];
            for (int i = 0; i < _shape[0]; i++)
            {
                for (int j = 0; j < _shape[1]; j++)
                {
                    for (int k = 0; k < _shape[2]; k++)
                    {
                        result[i, j, k] = _data[i * _strides[0] + j * _strides[1] + k];
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Get/set element at the given indices.
        /// </summary>
        public float this[params int[] indices]
        {
            get
            {
                ValidateIndices(indices);
                return _data[ComputeFlatIndex(indices)];
            }
            set
            {
                ValidateIndices(indices);
                _data[ComputeFlatIndex(indices)] = value;
            }
        }

        /// <summary>
        /// Reshape tensor to new shape (must have same total size).
        /// </summary>
        public Tensor Reshape(int[] newShape)
        {
            int newSize = ComputeSize(newShape);
            if (newSize != Size)
                throw new ArgumentException($"Cannot reshape tensor of size {Size} to size {newSize}");

            return new Tensor((float[])_data.Clone(), newShape);
        }

        /// <summary>
        /// Transpose two dimensions.
        /// </summary>
        public Tensor Transpose(int dim1, int dim2)
        {
            if (dim1 < 0 || dim1 >= Rank || dim2 < 0 || dim2 >= Rank)
                throw new ArgumentException($"Invalid dimensions for transpose: {dim1}, {dim2}");

            if (dim1 == dim2)
                return Clone();

            // Create new shape with swapped dimensions
            int[] newShape = (int[])_shape.Clone();
            newShape[dim1] = _shape[dim2];
            newShape[dim2] = _shape[dim1];

            Tensor result = new Tensor(newShape);

            // Transpose data
            int[] indices = new int[Rank];
            int[] newIndices = new int[Rank];

            for (int i = 0; i < Size; i++)
            {
                // Convert flat index to multi-dimensional indices
                int remaining = i;
                for (int d = 0; d < Rank; d++)
                {
                    indices[d] = remaining / _strides[d];
                    remaining = remaining % _strides[d];
                }

                // Swap the specified dimensions
                Array.Copy(indices, newIndices, Rank);
                newIndices[dim1] = indices[dim2];
                newIndices[dim2] = indices[dim1];

                result[newIndices] = _data[i];
            }

            return result;
        }

        /// <summary>
        /// Create a view/slice of the tensor along the first dimension.
        /// </summary>
        public Tensor Slice(int start, int length)
        {
            if (start < 0 || start + length > _shape[0])
                throw new ArgumentException($"Invalid slice: start={start}, length={length}, dim0={_shape[0]}");

            int[] newShape = (int[])_shape.Clone();
            newShape[0] = length;

            int elementsPerSlice = _strides[0];
            float[] newData = new float[length * elementsPerSlice];
            Array.Copy(_data, start * elementsPerSlice, newData, 0, newData.Length);

            return new Tensor(newData, newShape);
        }

        /// <summary>
        /// Get a single element along the first dimension as a new tensor.
        /// </summary>
        public Tensor GetRow(int index)
        {
            if (index < 0 || index >= _shape[0])
                throw new ArgumentException($"Invalid index {index} for dimension 0 of size {_shape[0]}");

            int[] newShape = new int[Rank - 1];
            Array.Copy(_shape, 1, newShape, 0, Rank - 1);

            int elementsPerRow = _strides[0];
            float[] newData = new float[elementsPerRow];
            Array.Copy(_data, index * elementsPerRow, newData, 0, elementsPerRow);

            return new Tensor(newData, newShape);
        }

        /// <summary>
        /// Create a deep copy of this tensor.
        /// </summary>
        public Tensor Clone()
        {
            return new Tensor((float[])_data.Clone(), _shape);
        }

        /// <summary>
        /// Fill tensor with a single value.
        /// </summary>
        public void Fill(float value)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] = value;
            }
        }

        /// <summary>
        /// Create a tensor filled with zeros.
        /// </summary>
        public static Tensor Zeros(int[] shape)
        {
            return new Tensor(shape);
        }

        /// <summary>
        /// Create a tensor filled with ones.
        /// </summary>
        public static Tensor Ones(int[] shape)
        {
            Tensor t = new Tensor(shape);
            t.Fill(1.0f);
            return t;
        }

        /// <summary>
        /// Create a tensor filled with random values from uniform distribution [0, 1).
        /// </summary>
        public static Tensor Random(int[] shape, Random rng = null)
        {
            rng = rng ?? new Random();
            Tensor t = new Tensor(shape);
            for (int i = 0; i < t.Size; i++)
            {
                t._data[i] = (float)rng.NextDouble();
            }
            return t;
        }

        /// <summary>
        /// Create a tensor filled with random values from normal distribution.
        /// </summary>
        public static Tensor RandomNormal(int[] shape, float mean = 0f, float std = 1f, Random rng = null)
        {
            rng = rng ?? new Random();
            Tensor t = new Tensor(shape);
            for (int i = 0; i < t.Size; i++)
            {
                // Box-Muller transform
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                t._data[i] = (float)(mean + std * normal);
            }
            return t;
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

        private static int ComputeSize(int[] shape)
        {
            int size = 1;
            foreach (int dim in shape)
            {
                if (dim <= 0)
                    throw new ArgumentException($"Invalid dimension size: {dim}");
                size *= dim;
            }
            return size;
        }

        private void ValidateIndices(int[] indices)
        {
            if (indices.Length != Rank)
                throw new ArgumentException($"Expected {Rank} indices, got {indices.Length}");

            for (int i = 0; i < Rank; i++)
            {
                if (indices[i] < 0 || indices[i] >= _shape[i])
                    throw new IndexOutOfRangeException($"Index {indices[i]} out of range for dimension {i} of size {_shape[i]}");
            }
        }

        private int ComputeFlatIndex(int[] indices)
        {
            int index = 0;
            for (int i = 0; i < Rank; i++)
            {
                index += indices[i] * _strides[i];
            }
            return index;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append($"Tensor(shape=[{string.Join(", ", _shape)}], data=");

            if (Size <= 20)
            {
                sb.Append("[");
                sb.Append(string.Join(", ", _data));
                sb.Append("]");
            }
            else
            {
                sb.Append($"[{_data[0]}, {_data[1]}, ... {_data[Size - 2]}, {_data[Size - 1]}]");
            }

            sb.Append(")");
            return sb.ToString();
        }
    }
}
