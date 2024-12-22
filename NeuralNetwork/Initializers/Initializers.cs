using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Initializers
{
    public static class Initializers
    {
        public static float[,] GlorotUniform(int inputDim, int outputDim)
        {
            var random = new Random();
            float[,] weights = new float[inputDim, outputDim];
            float limit = (float)Math.Sqrt(6.0 / (inputDim + outputDim));
            for (int i = 0; i < inputDim; i++)
            {
                for (int j = 0; j < outputDim; j++)
                {
                    weights[i, j] = (random.NextSingle() * 2 * limit) - limit;
                }
            }
            return weights;
        }
    }
}
