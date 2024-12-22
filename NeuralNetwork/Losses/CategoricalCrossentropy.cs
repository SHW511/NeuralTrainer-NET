using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Losses
{
    public class CategoricalCrossentropy : Loss
    {
        public override float Calculate(float[,] predicted, float[,] actual)
        {
            float loss = 0.0f;
            int samples = actual.GetLength(0);
            int classes = actual.GetLength(1);

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < classes; j++)
                {
                    float p = Math.Max(predicted[i, j], float.Epsilon); // Ensure predicted value is not zero
                    loss += actual[i, j] * (float)Math.Log(p);
                }
            }

            return -loss / samples;
        }
    }
}
