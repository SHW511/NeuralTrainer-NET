using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class LearningRateScheduler
    {
        private float initialLearningRate;
        private float decayRate;
        private int decaySteps;

        public LearningRateScheduler(float initialLearningRate, float decayRate, int decaySteps)
        {
            this.initialLearningRate = initialLearningRate;
            this.decayRate = decayRate;
            this.decaySteps = decaySteps;
        }

        public float GetLearningRate(int epoch)
        {
            return initialLearningRate * (float)Math.Pow(decayRate, epoch / decaySteps);
        }
    }
}
