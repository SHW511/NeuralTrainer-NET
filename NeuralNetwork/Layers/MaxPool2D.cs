using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Layers
{
    public class MaxPool2D : Layer
    {
        public int PoolSize { get; private set; }
        public int Stride { get; private set; }

        private float[,,] _inputs; // Store inputs for backpropagation

        public MaxPool2D(int poolSize, int stride = 2)
        {
            PoolSize = poolSize;
            Stride = stride;
        }

        public override void Build(int[] inputShape)
        {
            Built = true;
        }

        public override float[,] Call(float[,] inputs)
        {
            int inputHeight = inputs.GetLength(0);
            int inputWidth = inputs.GetLength(1);
            int inputChannels = InputShape[2];

            // Initialize _inputs to store the input values for backpropagation
            _inputs = new float[inputHeight, inputWidth, inputChannels];
            for (int i = 0; i < inputHeight; i++)
            {
                for (int j = 0; j < inputWidth; j++)
                {
                    _inputs[i, j, 0] = inputs[i, j];
                }
            }

            int outputHeight = (inputHeight - PoolSize) / Stride + 1;
            int outputWidth = (inputWidth - PoolSize) / Stride + 1;
            float[,,] outputs = new float[outputHeight, outputWidth, inputChannels];

            // Perform max pooling
            for (int c = 0; c < inputChannels; c++)
            {
                for (int i = 0; i < outputHeight; i++)
                {
                    for (int j = 0; j < outputWidth; j++)
                    {
                        float max = float.MinValue;
                        for (int pi = 0; pi < PoolSize; pi++)
                        {
                            for (int pj = 0; pj < PoolSize; pj++)
                            {
                                int ni = i * Stride + pi;
                                int nj = j * Stride + pj;
                                if (ni < inputHeight && nj < inputWidth)
                                {
                                    max = Math.Max(max, _inputs[ni, nj, c]);
                                }
                            }
                        }
                        outputs[i, j, c] = max;
                    }
                }
            }

            // Flatten the output to fit the expected return type
            int outputRows = outputs.GetLength(0);
            int outputCols = outputs.GetLength(1) * outputs.GetLength(2);
            float[,] flattenedOutputs = new float[outputRows, outputCols];
            for (int i = 0; i < outputRows; i++)
            {
                for (int j = 0; j < outputCols; j++)
                {
                    flattenedOutputs[i, j] = outputs[i, j / inputChannels, j % inputChannels];
                }
            }

            return flattenedOutputs;
        }

        public override float[,,,] Call(float[,,,] inputs)
        {
            int batchSize = inputs.GetLength(0);
            int inputHeight = inputs.GetLength(1);
            int inputWidth = inputs.GetLength(2);
            int inputChannels = inputs.GetLength(3);

            // Initialize _inputs to store the input values for backpropagation
            _inputs = new float[inputHeight, inputWidth, inputChannels];
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < inputHeight; i++)
                {
                    for (int j = 0; j < inputWidth; j++)
                    {
                        for (int c = 0; c < inputChannels; c++)
                        {
                            _inputs[i, j, c] = inputs[b, i, j, c];
                        }
                    }
                }
            }

            int outputHeight = (inputHeight - PoolSize) / Stride + 1;
            int outputWidth = (inputWidth - PoolSize) / Stride + 1;
            float[,,,] outputs = new float[batchSize, outputHeight, outputWidth, inputChannels];

            // Perform max pooling
            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < inputChannels; c++)
                {
                    for (int i = 0; i < outputHeight; i++)
                    {
                        for (int j = 0; j < outputWidth; j++)
                        {
                            float max = float.MinValue;
                            for (int pi = 0; pi < PoolSize; pi++)
                            {
                                for (int pj = 0; pj < PoolSize; pj++)
                                {
                                    int ni = i * Stride + pi;
                                    int nj = j * Stride + pj;
                                    if (ni < inputHeight && nj < inputWidth)
                                    {
                                        max = Math.Max(max, _inputs[ni, nj, c]);
                                    }
                                }
                            }
                            outputs[b, i, j, c] = max;
                        }
                    }
                }
            }

            return outputs;
        }

        public override float[,] Backward(float[,] gradient)
        {
            int inputHeight = _inputs.GetLength(0);
            int inputWidth = _inputs.GetLength(1);
            int inputChannels = _inputs.GetLength(2);

            int outputHeight = gradient.GetLength(0);
            int outputWidth = gradient.GetLength(1) / inputChannels;

            float[,,] gradient3D = new float[outputHeight, outputWidth, inputChannels];
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    for (int c = 0; c < inputChannels; c++)
                    {
                        gradient3D[i, j, c] = gradient[i, j * inputChannels + c];
                    }
                }
            }

            float[,,] inputGradient = new float[inputHeight, inputWidth, inputChannels];

            // Compute gradients
            for (int c = 0; c < inputChannels; c++)
            {
                for (int i = 0; i < outputHeight; i++)
                {
                    for (int j = 0; j < outputWidth; j++)
                    {
                        float max = float.MinValue;
                        int maxNi = -1, maxNj = -1;
                        for (int pi = 0; pi < PoolSize; pi++)
                        {
                            for (int pj = 0; pj < PoolSize; pj++)
                            {
                                int ni = i * Stride + pi;
                                int nj = j * Stride + pj;
                                if (ni < inputHeight && nj < inputWidth)
                                {
                                    if (_inputs[ni, nj, c] > max)
                                    {
                                        max = _inputs[ni, nj, c];
                                        maxNi = ni;
                                        maxNj = nj;
                                    }
                                }
                            }
                        }
                        if (maxNi != -1 && maxNj != -1)
                        {
                            inputGradient[maxNi, maxNj, c] += gradient3D[i, j, c];
                        }
                    }
                }
            }

            // Flatten the input gradient to fit the expected return type
            float[,] flattenedInputGradient = new float[inputHeight, inputWidth * inputChannels];
            for (int i = 0; i < inputHeight; i++)
            {
                for (int j = 0; j < inputWidth * inputChannels; j++)
                {
                    flattenedInputGradient[i, j] = inputGradient[i, j / inputChannels, j % inputChannels];
                }
            }

            return flattenedInputGradient;
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            int batchSize = gradient.GetLength(0);
            int inputHeight = _inputs.GetLength(0);
            int inputWidth = _inputs.GetLength(1);
            int inputChannels = _inputs.GetLength(2);

            int outputHeight = gradient.GetLength(1);
            int outputWidth = gradient.GetLength(2);

            float[,,,] inputGradient = new float[batchSize, inputHeight, inputWidth, inputChannels];

            // Compute gradients
            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < inputChannels; c++)
                {
                    for (int i = 0; i < outputHeight; i++)
                    {
                        for (int j = 0; j < outputWidth; j++)
                        {
                            float max = float.MinValue;
                            int maxNi = -1, maxNj = -1;
                            for (int pi = 0; pi < PoolSize; pi++)
                            {
                                for (int pj = 0; pj < PoolSize; pj++)
                                {
                                    int ni = i * Stride + pi;
                                    int nj = j * Stride + pj;
                                    if (ni < inputHeight && nj < inputWidth)
                                    {
                                        if (_inputs[ni, nj, c] > max)
                                        {
                                            max = _inputs[ni, nj, c];
                                            maxNi = ni;
                                            maxNj = nj;
                                        }
                                    }
                                }
                            }
                            if (maxNi != -1 && maxNj != -1)
                            {
                                inputGradient[b, maxNi, maxNj, c] += gradient[b, i, j, c];
                            }
                        }
                    }
                }
            }

            return inputGradient;
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            int inputHeight = inputShape[0];
            int inputWidth = inputShape[1];
            int outputHeight = (inputHeight - PoolSize) / Stride + 1;
            int outputWidth = (inputWidth - PoolSize) / Stride + 1;
            return new int[] { outputHeight, outputWidth, inputShape[2] };
        }
    }
}
