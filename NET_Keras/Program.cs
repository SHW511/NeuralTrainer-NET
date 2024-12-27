using System.Reflection.Emit;
using System.Reflection;
using NeuralNetwork;
using NeuralNetwork.Layers;
using NeuralNetwork.Layers.Cuda;
using NeuralNetwork.Layers.Activations;
using NeuralNetwork.Losses;
using NeuralNetwork.Optimizers;
using NeuralNetwork.Inference;
using NeuralNetwork.Processing.Text;
using System.Diagnostics;
using ManagedCuda;
using System.IO.MemoryMappedFiles;

namespace NET_Keras
{
    class Program
    {
        static void Main()
        {
            TryLargeTextGeneration();
        }

        public static void TryNumbers()
        {
            // Create a Sequential model
            var model = new Sequential();

            // Add a single Dense layer
            model.Add(new Dense(1, activation: Activations.ReLU));

            // Create loss function and optimizer instances
            var lossFunction = new MeanSquaredError();
            var optimizer = new SGD(0.01f);

            // Compile the model
            model.Compile(lossFunction, optimizer);

            // Build the model
            model.Build(new int[] { -1, 2 }); // Use -1 to indicate any batch size

            // Dummy training data
            float[,] xTrain = new float[,] { { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f }, { 7.0f, 8.0f } };
            float[,] yTrain = new float[,] { { 3.0f }, { 7.0f }, { 11.0f }, { 15.0f } };

            // Train the model
            for (int epoch = 0; epoch < 25; epoch++)
            {
                float loss = model.TrainOnBatch(xTrain, yTrain);
                if (epoch % 100 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}, Loss: {loss}");
                }
            }

            // Predict
            float[,] xTest = new float[,] { { 9.0f, 10.0f } };
            float[,] yPred = model.Predict(xTest);

            // Print the prediction
            Console.WriteLine($"Prediction: {yPred[0, 0]}");
        }

        public static void TryTextGeneration()
        {
            string text = File.ReadAllText("training_text.txt");

            // Sample text data
            //string text = "This is a sample text for training the text generator model. This text is used to demonstrate the text generation.";

            // Tokenize the text
            var vocab = TextPreProcessing.Tokenize(text);
            var reverseVocab = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);

            // Convert text to sequences
            var sequences = TextPreProcessing.TextToSequences(text, vocab);

            // Define the maximum sequence length
            int maxLen = 50;
            sequences = TextPreProcessing.PadSequences(sequences, maxLen);

            // Create a Sequential model
            var model = new Sequential();

            // Add layers to the model
            model.Add(new EmbeddingCuda(vocab.Count, 200)); // Embedding layer
            model.Add(new LSTM(50)); // LSTM layer
            model.Add(new DenseCuda(vocab.Count, activation: ActivationsCuda.SoftMax)); // Output layer

            Console.WriteLine("Layers added.");

            // Create loss function and optimizer instances
            var lossFunction = new CategoricalCrossentropy();
            var optimizer = new Adam(0.001f);

            // Compile the model
            model.Compile(lossFunction, optimizer);

            Console.WriteLine("Model compiled");

            // Build the model
            model.Build(new int[] { -1, maxLen }); // Use -1 to indicate any batch size

            // Dummy training data (for demonstration purposes)
            float[,] xTrain = sequences;
            float[,] yTrain = new float[sequences.GetLength(0), vocab.Count];
            for (int i = 0; i < sequences.GetLength(0); i++)
            {
                int nextWordIndex = (i + 1) % sequences.GetLength(0);
                yTrain[i, (int)sequences[nextWordIndex, 0]] = 1.0f;
            }

            Console.WriteLine("Starting training...");

            #region Training code
            //Train the model in parallel
            //Parallel.For(0, 10000, (i) =>
            //{
            //    float loss = model.TrainOnBatch(xTrain, yTrain);
            //    Console.WriteLine($"Epoch {i}, Loss: {loss}");
            //});

            //Wait for the ThreadPool to finish
            //while (ThreadPool.PendingWorkItemCount > 0)
            //{
            //    //Every second, report the number of pending work items
            //    Console.WriteLine($"Pending work items: {ThreadPool.PendingWorkItemCount}");

            //    Thread.Sleep(1000);
            //} 
            #endregion

            var lrScheduler = new LearningRateScheduler(0.001f, 0.96f, 1000);

            using (var context = new CudaContext())
            {
                var xTrainDevice = new CudaDeviceVariable<float>(xTrain.Length);
                var yTrainDevice = new CudaDeviceVariable<float>(yTrain.Length);

                xTrainDevice.CopyToDevice(xTrain);
                yTrainDevice.CopyToDevice(yTrain);

                // Train the model
                var stopwatch = new Stopwatch();
                stopwatch.Start();
                for (int epoch = 0; epoch < 100; epoch++)
                {
                    optimizer.learningRate = lrScheduler.GetLearningRate(epoch);
                    float loss = model.TrainOnBatch(xTrain, yTrain);

                    stopwatch.Stop();
                    Console.WriteLine($"Epoch {epoch}, Loss: {loss} Training time: {stopwatch.Elapsed}");
                    stopwatch.Restart();
                }

                stopwatch.Stop();
            }
            // Create a TextGenerator instance
            var textGenerator = new TextGenerator(model, reverseVocab, maxLen);

            // Generate text
            string seedText = "This is";
            int numWords = 10;
            string generatedText = textGenerator.GenerateText(seedText, numWords);

            // Print the generated text
            Console.WriteLine($"Generated Text: {generatedText}");
        }


        public static void TryLargeTextGeneration()
        {
            string text = File.ReadAllText("merged_clean.txt");

            // Tokenize the text
            var vocab = TextPreProcessing.Tokenize(text);
            var reverseVocab = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);

            // Convert text to sequences
            var sequences = TextPreProcessing.TextToSequences(text, vocab);

            // Define the maximum sequence length
            int maxLen = 50;
            sequences = TextPreProcessing.PadSequences(sequences, maxLen);

            // Reduce the size of xTrain and yTrain by using a subset of the sequences
            int subsetSize = 500; // Define a reasonable subset size
            subsetSize = Math.Min(subsetSize, sequences.GetLength(0)); // Ensure subset size does not exceed available sequences

            float[,] xTrain = new float[subsetSize, maxLen];
            Array.Copy(sequences, xTrain, subsetSize * maxLen);

            // Create a Sequential model
            var model = new Sequential();

            // Add layers to the model
            model.Add(new EmbeddingCuda(vocab.Count, 100)); // Embedding layer
            //model.Add(new LSTMCuda(50)); // LSTM layer
            model.Add(new DenseCuda(vocab.Count, activation: ActivationsCuda.SoftMax)); // Output layer

            Console.WriteLine("Layers added.");

            // Create loss function and optimizer instances
            var lossFunction = new CategoricalCrossentropy();
            var optimizer = new Adam(0.001f);

            // Compile the model
            model.Compile(lossFunction, optimizer);

            Console.WriteLine("Model compiled");

            // Build the model
            model.Build(new int[] { -1, maxLen }); // Use -1 to indicate any batch size

            // Process yTrain in smaller chunks
            int batchSize = 2500; // Define a reasonable batch size
            int numBatches = (int)Math.Ceiling((double)subsetSize / batchSize);

            Console.WriteLine("Starting training...");

            var lrScheduler = new LearningRateScheduler(0.001f, 0.96f, 1000);

            using (var context = new CudaContext())
            {
                var xTrainDevice = new CudaDeviceVariable<float>(xTrain.Length);
                xTrainDevice.CopyToDevice(xTrain);

                // Train the model in batches
                var stopwatch = new Stopwatch();
                stopwatch.Start();
                float meanLoss = 0.0f;
                for (int epoch = 0; epoch < 100; epoch++)
                {
                    optimizer.learningRate = lrScheduler.GetLearningRate(epoch);

                    for (int batch = 0; batch < numBatches; batch++)
                    {
                        int startIdx = batch * batchSize;
                        int endIdx = Math.Min(startIdx + batchSize, subsetSize);

                        // Create a smaller yTrain batch
                        float[,] yTrainBatch = new float[endIdx - startIdx, vocab.Count];
                        for (int i = startIdx; i < endIdx; i++)
                        {
                            int nextWordIndex = (i + 1) % subsetSize;
                            yTrainBatch[i - startIdx, (int)sequences[nextWordIndex, 0]] = 1.0f;
                        }

                        // Copy yTrainBatch to device
                        var yTrainDevice = new CudaDeviceVariable<float>(yTrainBatch.Length);
                        yTrainDevice.CopyToDevice(yTrainBatch);

                        // Train on the batch
                        float loss = model.TrainOnBatch(xTrain, yTrainBatch);
                        meanLoss += loss;
                        yTrainDevice.Dispose();
                    }

                    meanLoss /= numBatches;

                    stopwatch.Stop();
                    Console.WriteLine($"Epoch {epoch}, Loss: {meanLoss:00.00} Training time: {stopwatch.Elapsed}");
                    stopwatch.Restart();
                }

                stopwatch.Stop();
            }

            // Create a TextGenerator instance
            var textGenerator = new TextGenerator(model, reverseVocab, maxLen);

            // Generate text
            string seedText = "This is";
            int numWords = 10;
            string generatedText = textGenerator.GenerateText(seedText, numWords);

            // Print the generated text
            Console.WriteLine($"Generated Text: {generatedText}");
        }

    }
}
