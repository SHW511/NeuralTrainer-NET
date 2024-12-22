using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Optimizers
{
    public class Adam : Optimizer
    {
        public float learningRate;
        private float beta1;
        private float beta2;
        private List<float[,]> m;
        private List<float[,]> v;
        private int t;

        public Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f)
        {
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.m = new List<float[,]>();
            this.v = new List<float[,]>();
        }

        public void Initialize(List<float[,]> weights)
        {
            m.Clear();
            v.Clear();
            foreach (var weight in weights)
            {
                if (weight != null)
                {
                    m.Add(new float[weight.GetLength(0), weight.GetLength(1)]);
                    v.Add(new float[weight.GetLength(0), weight.GetLength(1)]);
                }
                else
                {
                    m.Add(new float[0, 0]); // Placeholder for null
                    v.Add(new float[0, 0]); // Placeholder for null
                }
            }
            t = 0;
        }

        public override void Update(List<float[,]> weights, List<float[,]> gradients)
        {
            if (m.Count == 0 || v.Count == 0)
            {
                Initialize(weights);
            }

            for (int i = 0; i < weights.Count; i++)
            {
                if (weights[i] != null && gradients[i] != null)
                {
                    weights[i] = Update(weights[i], gradients[i], i);
                }
            }
        }

        private float[,] Update(float[,] weights, float[,] gradients, int index)
        {
            t++;
            float[,] newWeights = new float[weights.GetLength(0), weights.GetLength(1)];

            int minRows = Math.Min(weights.GetLength(0), gradients.GetLength(0));
            int minCols = Math.Min(weights.GetLength(1), gradients.GetLength(1));

            for (int i = 0; i < minRows; i++)
            {
                for (int j = 0; j < minCols; j++)
                {
                    m[index][i, j] = beta1 * m[index][i, j] + (1 - beta1) * gradients[i, j];
                    v[index][i, j] = beta2 * v[index][i, j] + (1 - beta2) * (float)Math.Pow(gradients[i, j], 2);
                    float mHat = m[index][i, j] / (1 - (float)Math.Pow(beta1, t));
                    float vHat = v[index][i, j] / (1 - (float)Math.Pow(beta2, t));
                    newWeights[i, j] = weights[i, j] - learningRate * mHat / ((float)Math.Sqrt(vHat) + float.Epsilon);
                }
            }

            return newWeights;
        }
    }
}
