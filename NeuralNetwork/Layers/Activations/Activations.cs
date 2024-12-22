using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Layers.Activations
{
    public static class Activations
    {
        public static float[,] Linear(float[,] inputs)
        {
            // Create linear activation function
            return inputs;
        }

        // Overload without default parameters
        public static float[,] ReLU(float[,] inputs)
        {
            return ReLU(inputs, float.MaxValue, 0, 0);
        }

        // Original method with default parameters
        public static float[,] ReLU(float[,] inputs, float max_value = float.MaxValue, float threshold = 0, float negative_slope = 0)
        {
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            float[,] outputs = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float x = inputs[i, j];
                    if (x >= max_value)
                    {
                        outputs[i, j] = max_value;
                    }
                    else if (x >= threshold)
                    {
                        outputs[i, j] = x;
                    }
                    else
                    {
                        outputs[i, j] = negative_slope * (x - threshold);
                    }
                }
            }

            return outputs;
        }

        public static float[,] Sigmoid(float[,] inputs)
        {
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            float[,] outputs = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float x = inputs[i, j];
                    outputs[i, j] = (float)(1 / (1 + Math.Exp(-x)));
                }
            }

            return outputs;
        }

        public static float[,] SoftMax(float[,] inputs)
        {
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            float[,] outputs = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                float max = inputs.Cast<float>().Max();
                float sum = 0;
                for (int j = 0; j < cols; j++)
                {
                    outputs[i, j] = (float)Math.Exp(inputs[i, j] - max);
                    sum += outputs[i, j];
                }
                for (int j = 0; j < cols; j++)
                {
                    outputs[i, j] /= sum;
                }
            }

            return outputs;
        }
    }
}
