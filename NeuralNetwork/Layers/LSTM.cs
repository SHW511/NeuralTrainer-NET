using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Initializers;

namespace NeuralNetwork.Layers
{
    public class LSTM : Layer
    {
        public int Units { get; private set; }

        public float[,] _w;  // Weights and biases
        public float[,] _u;  // Weights and biases
        private float[] b;

        public LSTM(int units)
        {
            Units = units;
            _w = new float[0, 0]; // Initialize to avoid non-nullable warnings
            _u = new float[0, 0]; // Initialize to avoid non-nullable warnings
            b = new float[0];    // Initialize to avoid non-nullable warnings
        }

        public override void Build(int[] inputShape)
        {
            int inputDim = inputShape[1];
            _w = Initializers.Initializers.GlorotUniform(inputDim, Units * 4);
            _u = Initializers.Initializers.GlorotUniform(Units, Units * 4);

            b = new float[Units * 4];
            Built = true;
        }

        public override float[,] Call(float[,] inputs)
        {
            int timesteps = inputs.GetLength(0);
            int inputDim = inputs.GetLength(1);

            float[,] h = new float[timesteps, Units];
            float[] c = new float[Units]; // Cell state
            float[] h_t = new float[Units]; // Hidden state

            for (int t = 0; t < timesteps; t++)
            {
                float[] x_t = GetRow(inputs, t);
                float[] z = new float[Units * 4];

                for (int i = 0; i < Units * 4; i++)
                {
                    z[i] = DotProduct(x_t, GetColumn(_w, i)) + DotProduct(h_t, GetColumn(_u, i)) + b[i];
                }

                float[] i_gate = Sigmoid(GetSubArray(z, 0, Units));
                float[] f_gate = Sigmoid(GetSubArray(z, Units, Units));
                float[] o_gate = Sigmoid(GetSubArray(z, Units * 2, Units));
                float[] g_gate = Tanh(GetSubArray(z, Units * 3, Units));

                for (int i = 0; i < Units - 1; i++)
                {
                    c[i] = f_gate[i] * c[i] + i_gate[i] * g_gate[i];
                    h_t[i] = (float)(o_gate[i] * Math.Tanh(c[i])); // Use Math.Tanh for single float value
                }

                SetRow(h, t, h_t);
            }
            return h;
        }

        private float[] GetRow(float[,] matrix, int row)
        {
            int cols = matrix.GetLength(1);
            float[] result = new float[cols];
            for (int i = 0; i < cols; i++)
            {
                result[i] = matrix[row, i];
            }
            return result;
        }

        private float[] GetColumn(float[,] matrix, int col)
        {
            int rows = matrix.GetLength(0);
            float[] result = new float[rows];
            for (int i = 0; i < rows; i++)
            {
                result[i] = matrix[i, col];
            }
            return result;
        }

        private float DotProduct(float[] a, float[] b)
        {
            float result = 0;
            for (int i = 0; i < a.Length; i++)
            {
                result += a[i] * b[i];
            }
            return result;
        }

        private float[] GetSubArray(float[] array, int start, int length)
        {
            float[] result = new float[length];
            Array.Copy(array, start, result, 0, length);
            return result;
        }

        private float[] Sigmoid(float[] x)
        {
            float[] result = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                result[i] = (float)(1.0f / (1.0f + Math.Exp(-x[i])));
            }
            return result;
        }

        private float[] Tanh(float[] x)
        {
            float[] result = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                result[i] = (float)Math.Tanh(x[i]);
            }
            return result;
        }

        private void SetRow(float[,] matrix, int row, float[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                matrix[row, i] = values[i];
            }
        }

        public override float[,] Backward(float[,] gradient)
        {
            int timesteps = gradient.GetLength(0);
            int inputDim = _w.GetLength(0);

            float[,] dW = new float[inputDim, Units * 4];
            float[,] dU = new float[Units, Units * 4];
            float[] db = new float[Units * 4];
            float[,] dX = new float[timesteps, inputDim];

            float[] dh_next = new float[Units];
            float[] dc_next = new float[Units];

            // Initialize cell state and hidden state arrays for backpropagation
            float[,] c_states = new float[timesteps, Units];
            float[,] h_states = new float[timesteps, Units];

            // Forward pass to store cell states and hidden states
            float[] c = new float[Units];
            float[] h_t = new float[Units];
            for (int t = 0; t < timesteps; t++)
            {
                float[] x_t = GetRow(gradient, t);
                float[] z = new float[Units * 4];

                for (int i = 0; i < Units * 4; i++)
                {
                    z[i] = DotProduct(x_t, GetColumn(_w, i)) + DotProduct(h_t, GetColumn(_u, i)) + b[i];
                }

                float[] i_gate = Sigmoid(GetSubArray(z, 0, Units));
                float[] f_gate = Sigmoid(GetSubArray(z, Units, Units));
                float[] o_gate = Sigmoid(GetSubArray(z, Units * 2, Units));
                float[] g_gate = Tanh(GetSubArray(z, Units * 3, Units));

                for (int i = 0; i < Units - 1; i++)
                {
                    c[i] = f_gate[i] * c[i] + i_gate[i] * g_gate[i];
                    h_t[i] = (float)(o_gate[i] * Math.Tanh(c[i]));
                }

                SetRow(c_states, t, c);
                SetRow(h_states, t, h_t);
            }

            // Backward pass
            for (int t = timesteps - 1; t >= 0; t--)
            {
                float[] dh = GetRow(gradient, t);
                for (int i = 0; i < Units - 1; i++)
                {
                    dh[i] += dh_next[i];
                }

                float[] x_t = GetRow(dX, t);
                float[] z = new float[Units * 4];

                for (int i = 0; i < Units * 4; i++)
                {
                    z[i] = DotProduct(x_t, GetColumn(_w, i)) + DotProduct(dh, GetColumn(_u, i)) + b[i];
                }

                float[] i_gate = Sigmoid(GetSubArray(z, 0, Units));
                float[] f_gate = Sigmoid(GetSubArray(z, Units, Units));
                float[] o_gate = Sigmoid(GetSubArray(z, Units * 2, Units));
                float[] g_gate = Tanh(GetSubArray(z, Units * 3, Units));

                float[] c_t = GetRow(c_states, t);
                float[] dc = new float[Units];
                for (int i = 0; i < Units - 1; i++)
                {
                    dc[i] = (float)(dc_next[i] * f_gate[i] + dh[i] * o_gate[i] * (1 - Math.Tanh(c_t[i]) * Math.Tanh(c_t[i])));
                }

                for (int i = 0; i < Units - 1; i++)
                {
                    db[i] += dc[i];
                    for (int j = 0; j < inputDim; j++)
                    {
                        dW[j, i] += x_t[j] * dc[i];
                    }
                    for (int j = 0; j < Units - 1; j++)
                    {
                        dU[j, i] += dh[j] * dc[i];
                    }
                }

                for (int i = 0; i < Units - 1; i++)
                {
                    dh_next[i] = dc[i];
                    dc_next[i] = dc[i];
                }
            }

            // Update weights and biases
            //for (int i = 0; i < Units * 4; i++)
            //{
            //    b[i] -= 0.01f * db[i]; // Example learning rate
            //    for (int j = 0; j < inputDim; j++)
            //    {
            //        _w[j, i] -= 0.01f * dW[j, i]; // Example learning rate
            //    }
            //    for (int j = 0; j < Units - 1; j++)
            //    {
            //        _u[j, i] -= 0.01f * dU[j, i]; // Example learning rate
            //    }
            //}

            return dX;
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return new int[] { inputShape[0], Units };
        }
    }
}
