using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Optimizers
{
    [Serializable]
    public class SGD : Optimizer
    {
        public float LearningRate { get => learningRate; set => learningRate = value; }
        private float learningRate;

        public SGD()
        {
            
        }

        public SGD(float learningRate)
        {
            this.LearningRate = learningRate;
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
                                weights[i][j, k] -= LearningRate * gradients[i][j, k];
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

        public override void Update4D(List<float[,,,]> weights, List<float[,,,]> gradients)
        {
            throw new NotImplementedException();
        }
    }
}
