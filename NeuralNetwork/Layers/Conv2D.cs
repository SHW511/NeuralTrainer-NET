using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Layers
{
    public class Conv2D : Layer
    {
        public int Filters { get; private set; }
        public int KernelSize { get; private set; }
        public int Stride { get; private set; }
        public int Padding { get; private set; }
        public float[,,] Weights { get; private set; }
        public float[] Biases { get; private set; }

        private float[,,] _inputs; // Store inputs for backpropagation

        public Conv2D(int filters, int kernelSize, int stride = 1, int padding = 0)
        {
            Filters = filters;
            KernelSize = kernelSize;
            Stride = stride;
            Padding = padding;
        }

        public override void Build(int[] inputShape)
        {
            int inputChannels = inputShape[2];
            Weights = new float[Filters, inputChannels, KernelSize * KernelSize];
            Biases = new float[Filters];
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

            int outputHeight = (inputHeight - KernelSize + 2 * Padding) / Stride + 1;
            int outputWidth = (inputWidth - KernelSize + 2 * Padding) / Stride + 1;
            float[,,] outputs = new float[outputHeight, outputWidth, Filters];

            // Perform convolution
            for (int f = 0; f < Filters; f++)
            {
                for (int i = 0; i < outputHeight; i++)
                {
                    for (int j = 0; j < outputWidth; j++)
                    {
                        float sum = 0;
                        for (int c = 0; c < inputChannels; c++)
                        {
                            for (int ki = 0; ki < KernelSize; ki++)
                            {
                                for (int kj = 0; kj < KernelSize; kj++)
                                {
                                    int ni = i * Stride + ki - Padding;
                                    int nj = j * Stride + kj - Padding;
                                    if (ni >= 0 && ni < inputHeight && nj >= 0 && nj < inputWidth)
                                    {
                                        sum += _inputs[ni, nj, c] * Weights[f, c, ki * KernelSize + kj];
                                    }
                                }
                            }
                        }
                        outputs[i, j, f] = sum + Biases[f];
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
                    flattenedOutputs[i, j] = outputs[i, j / Filters, j % Filters];
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

            int outputHeight = (inputHeight - KernelSize + 2 * Padding) / Stride + 1;
            int outputWidth = (inputWidth - KernelSize + 2 * Padding) / Stride + 1;
            float[,,,] outputs = new float[batchSize, outputHeight, outputWidth, Filters];

            // Perform convolution
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < Filters; f++)
                {
                    for (int i = 0; i < outputHeight; i++)
                    {
                        for (int j = 0; j < outputWidth; j++)
                        {
                            float sum = 0;
                            for (int c = 0; c < inputChannels; c++)
                            {
                                for (int ki = 0; ki < KernelSize; ki++)
                                {
                                    for (int kj = 0; kj < KernelSize; kj++)
                                    {
                                        int ni = i * Stride + ki - Padding;
                                        int nj = j * Stride + kj - Padding;
                                        if (ni >= 0 && ni < inputHeight && nj >= 0 && nj < inputWidth)
                                        {
                                            sum += _inputs[ni, nj, c] * Weights[f, c, ki * KernelSize + kj];
                                        }
                                    }
                                }
                            }
                            outputs[b, i, j, f] = sum + Biases[f];
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
            int outputWidth = gradient.GetLength(1) / Filters;

            float[,,] gradient3D = new float[outputHeight, outputWidth, Filters];
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    for (int f = 0; f < Filters; f++)
                    {
                        gradient3D[i, j, f] = gradient[i, j * Filters + f];
                    }
                }
            }

            float[,,] inputGradient = new float[inputHeight, inputWidth, inputChannels];
            float[,,] weightGradient = new float[Filters, inputChannels, KernelSize * KernelSize];
            float[] biasGradient = new float[Filters];

            // Compute gradients
            for (int f = 0; f < Filters; f++)
            {
                for (int i = 0; i < outputHeight; i++)
                {
                    for (int j = 0; j < outputWidth; j++)
                    {
                        biasGradient[f] += gradient3D[i, j, f];
                        for (int c = 0; c < inputChannels; c++)
                        {
                            for (int ki = 0; ki < KernelSize; ki++)
                            {
                                for (int kj = 0; kj < KernelSize; kj++)
                                {
                                    int ni = i * Stride + ki - Padding;
                                    int nj = j * Stride + kj - Padding;
                                    if (ni >= 0 && ni < inputHeight && nj >= 0 && nj < inputWidth)
                                    {
                                        weightGradient[f, c, ki * KernelSize + kj] += _inputs[ni, nj, c] * gradient3D[i, j, f];
                                        inputGradient[ni, nj, c] += Weights[f, c, ki * KernelSize + kj] * gradient3D[i, j, f];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Update weights and biases
            for (int f = 0; f < Filters; f++)
            {
                Biases[f] -= 0.01f * biasGradient[f]; // Example learning rate
                for (int c = 0; c < inputChannels; c++)
                {
                    for (int k = 0; k < KernelSize * KernelSize; k++)
                    {
                        Weights[f, c, k] -= 0.01f * weightGradient[f, c, k]; // Example learning rate
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
            float[,,] weightGradient = new float[Filters, inputChannels, KernelSize * KernelSize];
            float[] biasGradient = new float[Filters];

            // Compute gradients
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < Filters; f++)
                {
                    for (int i = 0; i < outputHeight; i++)
                    {
                        for (int j = 0; j < outputWidth; j++)
                        {
                            biasGradient[f] += gradient[b, i, j, f];
                            for (int c = 0; c < inputChannels; c++)
                            {
                                for (int ki = 0; ki < KernelSize; ki++)
                                {
                                    for (int kj = 0; kj < KernelSize; kj++)
                                    {
                                        int ni = i * Stride + ki - Padding;
                                        int nj = j * Stride + kj - Padding;
                                        if (ni >= 0 && ni < inputHeight && nj >= 0 && nj < inputWidth)
                                        {
                                            weightGradient[f, c, ki * KernelSize + kj] += _inputs[ni, nj, c] * gradient[b, i, j, f];
                                            inputGradient[b, ni, nj, c] += Weights[f, c, ki * KernelSize + kj] * gradient[b, i, j, f];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Update weights and biases
            for (int f = 0; f < Filters; f++)
            {
                Biases[f] -= 0.01f * biasGradient[f]; // Example learning rate
                for (int c = 0; c < inputChannels; c++)
                {
                    for (int k = 0; k < KernelSize * KernelSize; k++)
                    {
                        Weights[f, c, k] -= 0.01f * weightGradient[f, c, k]; // Example learning rate
                    }
                }
            }

            return inputGradient;
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            int inputHeight = inputShape[0];
            int inputWidth = inputShape[1];
            int outputHeight = (inputHeight - KernelSize + 2 * Padding) / Stride + 1;
            int outputWidth = (inputWidth - KernelSize + 2 * Padding) / Stride + 1;
            return new int[] { outputHeight, outputWidth, Filters };
        }
    }
}
