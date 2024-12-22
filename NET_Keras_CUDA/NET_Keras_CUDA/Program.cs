using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Hybridizer.Runtime.CUDAImports;

namespace NET_Keras_CUDA
{
    internal class Program
    {
        [EntryPoint]
        public static void Run(int N, int[] a, int[] b)
        {
            Parallel.For(0, N, i => { a[i] += b[i]; });
        }

        static void Main(string[] args)
        {
            int[] a = { 1, 2, 3, 4, 5 };
            int[] b = { 10, 20, 30, 40, 50 };

            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            //if .SetDistrib is not used, the default is .SetDistrib(prop.multiProcessorCount * 16, 128)
            HybRunner runner = HybRunner.Cuda();

            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new Program());

            wrapped.Run(5, a, b);

            Console.Out.WriteLine("DONE");
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
            int maxLen = 100;
            sequences = TextPreProcessing.PadSequences(sequences, maxLen);

            // Create a Sequential model
            var model = new Sequential();

            // Add layers to the model
            model.Add(new Embedding(vocab.Count, 50)); // Embedding layer
            model.Add(new LSTM(50)); // LSTM layer
            model.Add(new Dense(vocab.Count, activation: Activations.SoftMax)); // Output layer

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

            // Train the model
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            for (int epoch = 0; epoch < 10000; epoch++)
            {
                float loss = model.TrainOnBatch(xTrain, yTrain);

                stopwatch.Stop();
                Console.WriteLine($"Epoch {epoch}, Loss: {loss} Training time: {stopwatch.Elapsed}");
                stopwatch.Restart();
            }
            stopwatch.Stop();

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
