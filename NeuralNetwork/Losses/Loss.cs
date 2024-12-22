using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Losses
{
    public abstract class Loss
    {
        public abstract float Calculate(float[,] predicted, float[,] actual);
    }
}
