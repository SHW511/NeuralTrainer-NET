using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Optimizers
{
    public class SGD : Optimizer
    {
        private float learningRate;

        public SGD(float learningRate)
        {
            this.learningRate = learningRate;
        }

        public override void Update(List<float[,]> weights, List<float[,]> gradients)
        {
            for (int i = 0; i < weights.Count; i++)
            {
                try
                {
                    for (int j = 0; j < weights[i].GetLength(0); j++)
                    {
                        try
                        {
                            for (int k = 0; k < weights[i].GetLength(1); k++)
                            {
                                weights[i][j, k] -= learningRate * gradients[i][j, k];
                            }
                        }
                        catch (IndexOutOfRangeException)
                        {
                            // Continue with the next iteration of the j loop
                            continue;
                        }
                    }
                }
                catch (IndexOutOfRangeException)
                {
                    // Continue with the next iteration of the i loop
                    continue;
                }
            }
        }
    }
}
