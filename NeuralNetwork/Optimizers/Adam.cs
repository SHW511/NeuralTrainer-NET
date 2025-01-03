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
        private List<float[,,,]> m4D;
        private List<float[,,,]> v4D;
        private int t;

        public Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f)
        {
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.m = new List<float[,]>();
            this.v = new List<float[,]>();
            this.m4D = new List<float[,,,]>();
            this.v4D = new List<float[,,,]>();
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

        public void Initialize4D(List<float[,,,]> weights)
        {
            m4D.Clear();
            v4D.Clear();
            foreach (var weight in weights)
            {
                if (weight != null)
                {
                    m4D.Add(new float[weight.GetLength(0), weight.GetLength(1), weight.GetLength(2), weight.GetLength(3)]);
                    v4D.Add(new float[weight.GetLength(0), weight.GetLength(1), weight.GetLength(2), weight.GetLength(3)]);
                }
                else
                {
                    m4D.Add(new float[0, 0, 0, 0]); // Placeholder for null
                    v4D.Add(new float[0, 0, 0, 0]); // Placeholder for null
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

        public override void Update4D(List<float[,,,]> weights, List<float[,,,]> gradients)
        {
            if (m4D.Count == 0 || v4D.Count == 0)
            {
                Initialize4D(weights);
            }

            for (int i = 0; i < weights.Count; i++)
            {
                if (weights[i] != null && gradients[i] != null)
                {
                    weights[i] = Update4D(weights[i], gradients[i], i);
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

        private float[,,,] Update4D(float[,,,] weights, float[,,,] gradients, int index)
        {
            t++;
            float[,,,] newWeights = new float[weights.GetLength(0), weights.GetLength(1), weights.GetLength(2), weights.GetLength(3)];

            int dim0 = Math.Min(weights.GetLength(0), gradients.GetLength(0));
            int dim1 = Math.Min(weights.GetLength(1), gradients.GetLength(1));
            int dim2 = Math.Min(weights.GetLength(2), gradients.GetLength(2));
            int dim3 = Math.Min(weights.GetLength(3), gradients.GetLength(3));

            for (int i = 0; i < dim0; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    for (int k = 0; k < dim2; k++)
                    {
                        for (int l = 0; l < dim3; l++)
                        {
                            m4D[index][i, j, k, l] = beta1 * m4D[index][i, j, k, l] + (1 - beta1) * gradients[i, j, k, l];
                            v4D[index][i, j, k, l] = beta2 * v4D[index][i, j, k, l] + (1 - beta2) * (float)Math.Pow(gradients[i, j, k, l], 2);
                            float mHat = m4D[index][i, j, k, l] / (1 - (float)Math.Pow(beta1, t));
                            float vHat = v4D[index][i, j, k, l] / (1 - (float)Math.Pow(beta2, t));
                            newWeights[i, j, k, l] = weights[i, j, k, l] - learningRate * mHat / ((float)Math.Sqrt(vHat) + float.Epsilon);
                        }
                    }
                }
            }

            return newWeights;
        }
    }
}
