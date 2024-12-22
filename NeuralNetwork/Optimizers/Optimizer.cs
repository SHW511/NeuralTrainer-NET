using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Optimizers
{
    public abstract class Optimizer
    {
        public abstract void Update(List<float[,]> weights, List<float[,]> gradients);
    }
}
