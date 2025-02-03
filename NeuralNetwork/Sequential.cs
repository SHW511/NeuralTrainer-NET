using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;
using NeuralNetwork.Losses;
using NeuralNetwork.Optimizers;
using NeuralNetwork.SerializationHelper;

namespace NeuralNetwork
{
    [Serializable]
    public class Sequential : ISerializable, IXmlSerializable
    {
        private List<Layer> layers;
        private bool built;
        private Loss? lossFunction; // Marked as nullable
        private Optimizer? optimizer; // Marked as nullable

        public List<Layer> Layers { get => layers; set => layers = value; }
        public bool Built { get => built; set => built = value; }
        public Loss LossFunction { get => lossFunction; set => lossFunction = value; }
        public Optimizer Optimizer { get => optimizer; set => optimizer = value; }

        public Sequential()
        {
            Layers = new List<Layer>();
            Built = false;
            LossFunction = null; // Initialize to null
            Optimizer = null; // Initialize to null
        }

        protected Sequential(SerializationInfo info, StreamingContext context)
        {
            Layers = (List<Layer>)info.GetValue("Layers", typeof(List<Layer>));
            Built = info.GetBoolean("Built");
            LossFunction = (Loss)info.GetValue("LossFunction", typeof(Loss));
            Optimizer = (Optimizer)info.GetValue("Optimizer", typeof(Optimizer));
        }

        public void Add(Layer layer)
        {
            Layers.Add(layer);
            if (!Built)
            {
                layer.Build(new int[] { 1, layer.InputDim });
                Built = true;
            }
            else
            {
                var prevLayerOutputShape = Layers[Layers.Count - 2].GetOutputShape(new int[] { -1, Layers[Layers.Count - 2].OutputDim });
                layer.Build(prevLayerOutputShape);
            }
        }

        public void Build(int[] inputShape)
        {
            int[] shape = inputShape;
            foreach (var layer in Layers)
            {
                layer.Build(shape);
                shape = layer.GetOutputShape(shape);
            }
        }

        public void Compile(Loss lossFunction, Optimizer optimizer)
        {
            this.LossFunction = lossFunction;
            this.Optimizer = optimizer;
        }

        public void Fit(float[,] x, float[,] y, int epochs = 1, int batchSize = 32, int reportInterval = 10)
        {
            if (LossFunction == null || Optimizer == null)
            {
                throw new InvalidOperationException("The model must be compiled before fitting.");
            }

            int numSamples = x.GetLength(0);

            Stopwatch stopwatch = new Stopwatch();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                stopwatch.Restart();

                float epochLoss = 0.0f;
                for (int i = 0; i < numSamples; i += batchSize)
                {
                    int batchEnd = Math.Min(i + batchSize, numSamples);
                    float[,] xBatch = GetBatch(x, i, batchEnd);
                    float[,] yBatch = GetBatch(y, i, batchEnd);
                    epochLoss += TrainOnBatch(xBatch, yBatch);
                    stopwatch.Stop();
                }
                if (epoch % reportInterval == 0)
                {
                    Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {epochLoss}, Time: {stopwatch.Elapsed}");
                }
            }
        }

        public void Fit4D(float[,,,] x, float[,] y, int epochs = 1, int batchSize = 32, int reportInterval = 10)
        {
            if (LossFunction == null || Optimizer == null)
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
                    float[,,,] xBatch = GetBatch4D(x, i, batchEnd);
                    float[,] yBatch = GetBatch(y, i, batchEnd);
                    epochLoss += TrainOnBatch4D(xBatch, yBatch);
                }
                if (epoch % reportInterval == 0)
                {
                    Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {epochLoss / (numSamples / batchSize)}");
                }
            }
        }

        private float[,,,] GetBatch4D(float[,,,] data, int start, int end)
        {
            int dim1 = data.GetLength(1);
            int dim2 = data.GetLength(2);
            int dim3 = data.GetLength(3);
            float[,,,] batch = new float[end - start, dim1, dim2, dim3];
            for (int i = start; i < end; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    for (int k = 0; k < dim2; k++)
                    {
                        for (int l = 0; l < dim3; l++)
                        {
                            batch[i - start, j, k, l] = data[i, j, k, l];
                        }
                    }
                }
            }
            return batch;
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
            if (LossFunction == null || Optimizer == null)
            {
                throw new InvalidOperationException("The model must be compiled before training.");
            }

            float[,] output = x;
            List<float[,]> activations = new List<float[,]>();

            // Forward pass
            foreach (var layer in Layers)
            {
                output = layer.Call(output);
                activations.Add(output);
            }

            float lossValue = LossFunction.Calculate(y, output);

            // Backward pass
            List<float[,]> gradients = new List<float[,]>();
            float[,] dLoss = ComputeLossGradient(y, output);

            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                var layer = Layers[i];
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
            List<float[,]> weights = Layers.Select(l => l.Weights).ToList();
            Optimizer.Update(weights, gradients);

            return lossValue;
        }

        public float TrainOnBatch4D(float[,,,] x, float[,] y)
        {
            if (LossFunction == null || Optimizer == null)
            {
                throw new InvalidOperationException("The model must be compiled before training.");
            }

            float[,,,] output = x;
            List<float[,,,]> activations = new List<float[,,,]>();

            // Forward pass
            foreach (var layer in Layers)
            {
                output = layer.Call(output);
                activations.Add(output);
            }

            float lossValue = LossFunction.Calculate4D(output, y);

            // Backward pass
            List<float[,,,]> gradients = new List<float[,,,]>();
            float[,,,] dLoss = ComputeLossGradient4D(y, output);

            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                var layer = Layers[i];
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
                        for (int l = 0; l < gradients[i].GetLength(2); l++)
                        {
                            for (int m = 0; m < gradients[i].GetLength(3); m++)
                            {
                                gradients[i][j, k, l, m] = Math.Max(Math.Min(gradients[i][j, k, l, m], clipValue), -clipValue);
                            }
                        }
                    }
                }
            }

            // Update weights
            List<float[,,,]> weights = Layers.Select(l => l.Weights).Cast<float[,,,]>().ToList();
            Optimizer.Update4D(weights, gradients);

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

        private float[,,,] ComputeLossGradient4D(float[,] y, float[,,,] output)
        {
            // Implement the gradient calculation for the loss function
            // This is a placeholder implementation
            int dim0 = y.GetLength(0);
            int dim1 = output.GetLength(1);
            int dim2 = output.GetLength(2);
            int dim3 = output.GetLength(3);
            float[,,,] gradient = new float[dim0, dim1, dim2, dim3];
            for (int i = 0; i < dim0; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    for (int k = 0; k < dim2; k++)
                    {
                        for (int l = 0; l < dim3; l++)
                        {
                            gradient[i, j, k, l] = output[i, j, k, l] - y[i, j];
                        }
                    }
                }
            }
            return gradient;
        }

        public float[,] Predict(float[,] x)
        {
            float[,] output = x;
            foreach (var layer in Layers)
            {
                output = layer.Call(output);
            }

            // Clip the output to avoid numerical instability
            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int j = 0; j < output.GetLength(1); j++)
                {
                    output[i, j] = Math.Max(Math.Min(output[i, j], 1.0f), float.Epsilon);
                }
            }

            return output;
        }

        public float[,,,] Predict4D(float[,,,] x)
        {
            float[,,,] output = x;
            foreach (var layer in Layers)
            {
                output = layer.Call(output);
            }
            return output;
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("Layers", Layers);
            info.AddValue("Built", Built);
            info.AddValue("LossFunction", LossFunction);
            info.AddValue("Optimizer", Optimizer);
        }

        public XmlSchema GetSchema()
        {
            throw new NotImplementedException();
        }

        public void ReadXml(XmlReader reader)
        {
            throw new NotImplementedException();
        }

        public void WriteXml(XmlWriter writer)
        {
            writer.WriteElementString("Built", Built.ToString());
            writer.WriteStartElement("Layers");
            foreach (var layer in Layers)
            {
                writer.WriteStartElement(layer.GetType().Name);

                var jaggedWeights = ArrayHelpers.ConvertToJaggedArray(layer.Weights);

                writer.WriteStartElement("Weights");
                for (int i = 0; i < jaggedWeights.Length; i++)
                {
                    for (int j = 0; j < jaggedWeights[i].Length; j++)
                    {
                        writer.WriteElementString("Weight", jaggedWeights[i][j].ToString());
                    }
                }
                writer.WriteEndElement();

                writer.WriteEndElement();
            }
            writer.WriteEndElement();
        }
    }
}
