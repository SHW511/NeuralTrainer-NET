using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Losses;
using NeuralNetwork.Optimizers;

namespace NeuralNetwork
{
    public class Sequential
    {
        private List<Layer> layers;
        private bool built;
        private Loss? lossFunction; // Marked as nullable
        private Optimizer? optimizer; // Marked as nullable

        public Sequential()
        {
            layers = new List<Layer>();
            built = false;
            lossFunction = null; // Initialize to null
            optimizer = null; // Initialize to null
        }

        public void Add(Layer layer)
        {
            layers.Add(layer);
            if (!built)
            {
                layer.Build(new int[] { -1, layer.InputDim });
                built = true;

            }
            else
            {
                var prevLayerOutputShape = layers[layers.Count - 2].GetOutputShape(new int[] { -1, layers[layers.Count - 2].OutputDim });
                layer.Build(prevLayerOutputShape);
            }
        }

        public void Build(int[] inputShape)
        {
            int[] shape = inputShape;
            foreach (var layer in layers)
            {
                layer.Build(shape);
                shape = layer.GetOutputShape(shape);
            }
        }

        public void Compile(Loss lossFunction, Optimizer optimizer)
        {
            this.lossFunction = lossFunction;
            this.optimizer = optimizer;
        }

        public void Fit(float[,] x, float[,] y, int epochs = 1, int batchSize = 32, int reportInterval = 10)
        {
            if (lossFunction == null || optimizer == null)
            {
                throw new InvalidOperationException("The model must be compiled before fitting.");
            }

            int numSamples = x.GetLength(0);
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float epochLoss = 0.0f;
                for (int i = 0; i < numSamples; i += batchSize)
                {
                    int batchEnd = Math.Min(i + batchSize, numSamples);
                    float[,] xBatch = GetBatch(x, i, batchEnd);
                    float[,] yBatch = GetBatch(y, i, batchEnd);
                    epochLoss += TrainOnBatch(xBatch, yBatch);
                }
                if (epoch % reportInterval == 0)
                {
                    Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {epochLoss / (numSamples / batchSize)}");
                }
            }
        }

        private float[,] GetBatch(float[,] data, int start, int end)
        {
            int cols = data.GetLength(1);
            float[,] batch = new float[end - start, cols];
            for (int i = start; i < end; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    batch[i - start, j] = data[i, j];
                }
            }
            return batch;
        }

        public float TrainOnBatch(float[,] x, float[,] y)
        {
            if (lossFunction == null || optimizer == null)
            {
                throw new InvalidOperationException("The model must be compiled before training.");
            }

            float[,] output = x;
            List<float[,]> activations = new List<float[,]>();

            // Forward pass
            foreach (var layer in layers)
            {
                output = layer.Call(output);
                activations.Add(output);
            }

            float lossValue = lossFunction.Calculate(y, output);

            // Backward pass
            List<float[,]> gradients = new List<float[,]>();
            float[,] dLoss = ComputeLossGradient(y, output);

            for (int i = layers.Count - 1; i >= 0; i--)
            {
                var layer = layers[i];
                dLoss = layer.Backward(dLoss);
                gradients.Add(dLoss);
            }

            gradients.Reverse();

            // Gradient clipping
            float clipValue = 1.0f;
            for (int i = 0; i < gradients.Count; i++)
            {
                if (gradients[i] == null)
                    continue;

                for (int j = 0; j < gradients[i].GetLength(0); j++)
                {
                    for (int k = 0; k < gradients[i].GetLength(1); k++)
                    {
                        gradients[i][j, k] = Math.Max(Math.Min(gradients[i][j, k], clipValue), -clipValue);
                    }
                }
            }

            // Update weights
            List<float[,]> weights = layers.Select(l => l.Weights).ToList();
            optimizer.Update(weights, gradients);

            return lossValue;
        }

        private float[,] ComputeLossGradient(float[,] y, float[,] output)
        {
            // Implement the gradient calculation for the loss function
            // This is a placeholder implementation
            float[,] gradient = new float[y.GetLength(0), y.GetLength(1)];
            for (int i = 0; i < y.GetLength(0); i++)
            {
                for (int j = 0; j < y.GetLength(1); j++)
                {
                    gradient[i, j] = output[i, j] - y[i, j];
                }
            }
            return gradient;
        }

        public float[,] Predict(float[,] x)
        {
            float[,] output = x;
            foreach (var layer in layers)
            {
                output = layer.Call(output);
            }
            return output;
        }
    }
}
