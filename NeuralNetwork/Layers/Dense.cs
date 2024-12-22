using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Initializers;
using NeuralNetwork.Layers.Activations;

namespace NeuralNetwork.Layers
{
    public class Dense : Layer
    {
        public int OutputDim { get; private set; }
        public Func<int, int, float[,]> Init { get; private set; }
        public Func<float[,], float[,]> Activation { get; private set; }
        public float[] Biases { get; private set; }
        public bool UseBias { get; private set; }
        public float[,] InitialWeights { get; private set; }

        private float[,] inputs; // Store inputs for backpropagation

        public Dense(int outputDim, Func<int, int, float[,]> init = null,
                             Func<float[,], float[,]> activation = null, float[,] weights = null,
                             bool useBias = true)
        {
            OutputDim = outputDim;
            Init = init ?? Initializers.Initializers.GlorotUniform;
            Activation = activation ?? Activations.Activations.Linear;
            InitialWeights = weights;
            UseBias = useBias;
        }

        public override void Build(int[] inputShape)
        {
            if (inputShape.Length != 2)
                throw new ArgumentException("Input shape should be a 2D tensor");

            InputShape = inputShape;
            Weights = Init(inputShape[1], OutputDim);
            if (UseBias)
            {
                Biases = new float[OutputDim];
            }

            // Apply initial weights if provided
            if (InitialWeights != null)
            {
                Weights = InitialWeights;
            }

            Built = true;
        }

        public override float[,] Call(float[,] inputs)
        {
            if (!Built)
            {
                // Determine the input shape from the inputs
                int[] inputShape = { inputs.GetLength(0), inputs.GetLength(1) };
                Build(inputShape);
            }

            this.inputs = inputs; // Store inputs for backpropagation

            var output = Dot(inputs, Weights);
            if (UseBias)
            {
                output = AddBias(output, Biases);
            }

            return Activation(output);
        }

        public override float[,] Backward(float[,] gradient)
        {
            if (InputShape == null || InputShape.Length != 2)
                throw new InvalidOperationException("InputShape must be set before calling Backward");

            int batchSize = inputs.GetLength(0);
            int inputDim = InputShape[1] - 1;
            float[,] weightGradient = new float[inputDim, OutputDim];
            float[] biasGradient = new float[OutputDim];
            float[,] inputGradient = new float[batchSize, inputDim];

            // Compute gradients with respect to weights and biases
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < OutputDim; j++)
                {
                    if (UseBias)
                    {
                        biasGradient[j] += gradient[i, j];
                    }
                    for (int k = 0; k < inputDim; k++)
                    {
                        weightGradient[k, j] += inputs[i, k] * gradient[i, j];
                        inputGradient[i, k] += Weights[k, j] * gradient[i, j];
                    }
                }
            }

            // Update weights and biases
            for (int j = 0; j < OutputDim; j++)
            {
                if (UseBias)
                {
                    Biases[j] -= 0.01f * biasGradient[j]; // Example learning rate
                }
                for (int k = 0; k < inputDim; k++)
                {
                    Weights[k, j] -= 0.01f * weightGradient[k, j]; // Example learning rate
                }
            }

            return inputGradient;
        }

        private float[,] Dot(float[,] a, float[,] b)
        {
            // Implementation of dot product
            int aRows = a.GetLength(0);
            int aCols = a.GetLength(1);
            int bCols = b.GetLength(1);
            float[,] result = new float[aRows, bCols];

            for (int i = 0; i < aRows; i++)
            {
                for (int j = 0; j < bCols; j++)
                {
                    result[i, j] = 0;
                    for (int k = 0; k < aCols; k++)
                    {
                        result[i, j] += a[i, k] * b[k, j];
                    }
                }
            }
            return result;
        }

        private float[,] AddBias(float[,] output, float[] bias)
        {
            int rows = output.GetLength(0);
            int cols = output.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    output[i, j] += bias[j];
                }
            }
            return output;
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            if (inputShape.Length != 2)
                throw new ArgumentException("Input shape should be a 2D tensor");

            return new int[] { inputShape[0], OutputDim };
        }

        public override Dictionary<string, object> GetConfig()
        {
            var config = new Dictionary<string, object>
                                {
                                    {"outputDim", OutputDim},
                                    {"init", Init.Method.Name},
                                    {"activation", Activation.Method.Name},
                                    {"useBias", UseBias}
                                };
            var baseConfig = base.GetConfig();
            foreach (var kv in baseConfig)
            {
                config[kv.Key] = kv.Value;
            }
            return config;
        }
    }
}
