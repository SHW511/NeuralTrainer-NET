using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Losses
{
    [Serializable]
    public class MeanSquaredError : Loss
    {
        public override float Calculate(float[,] truth, float[,] pred)
        {
            float loss = 0.0f;
            int rows = truth.GetLength(0);
            int cols = truth.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float diff = truth[i, j] - pred[i, j];
                    loss += diff * diff;
                }
            }
            return loss / (rows * cols);
        }

        public override float Calculate4D(float[,,,] predicted, float[,] actual)
        {
            throw new NotImplementedException();
        }
    }
}
