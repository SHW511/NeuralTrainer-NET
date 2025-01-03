using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Initializers;

namespace NeuralNetwork.Layers
{
    public class Embedding : Layer
    {
        private float[,] embeddings;
        private int[] inputIndices; // Store input indices for backpropagation

        public Embedding(int inputDim, int outputDim)
        {
            InputDim = inputDim;
            OutputDim = outputDim;
        }

        public override float[,] Backward(float[,] gradient)
        {
            int samples = gradient.GetLength(0);
            int sequenceLength = gradient.GetLength(1) / OutputDim;

            float[,] dEmbeddings = new float[InputDim, OutputDim];

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < sequenceLength; j++)
                {
                    int index = inputIndices[i * sequenceLength + j];
                    for (int k = 0; k < OutputDim; k++)
                    {
                        dEmbeddings[index, k] += gradient[i, j * OutputDim + k];
                    }
                }
            }

            // Update embeddings
            for (int i = 0; i < InputDim; i++)
            {
                for (int j = 0; j < OutputDim; j++)
                {
                    embeddings[i, j] -= 0.01f * dEmbeddings[i, j]; // Example learning rate
                }
            }

            return null; // Embedding layer does not propagate gradients to previous layers
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException();
        }

        public override void Build(int[] inputShape)
        {
            embeddings = Initializers.Initializers.GlorotUniform(InputDim, OutputDim);
            Built = true;
        }

        public override float[,] Call(float[,] inputs)
        {
            int samples = inputs.GetLength(0);
            int sequenceLength = inputs.GetLength(1);
            float[,] output = new float[samples, sequenceLength * OutputDim];
            inputIndices = new int[samples * sequenceLength];

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < sequenceLength; j++)
                {
                    int index = (int)inputs[i, j];
                    inputIndices[i * sequenceLength + j] = index;
                    for (int k = 0; k < OutputDim; k++)
                    {
                        output[i, j * OutputDim + k] = embeddings[index, k];
                    }
                }
            }
            return output;
        }

        public override float[,,,] Call(float[,,,] inputs)
        {
            throw new NotImplementedException();
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return new int[] { inputShape[0], inputShape[1] * OutputDim };
        }
    }
}
