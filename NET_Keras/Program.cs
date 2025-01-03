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
using Newtonsoft.Json;
using System.Drawing;
using System.Formats.Asn1;

namespace NET_Keras
{
    class Program
    {
        static void Main()
        {
            TryTextGeneration();
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

            // Tokenize the text
            var vocab = TextPreProcessing.Tokenize(text);
            var reverseVocab = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);

            // Convert text to sequences
            var sequences = TextPreProcessing.TextToSequences(text, vocab);

            // Define the maximum sequence length
            int maxLen = 150;
            sequences = TextPreProcessing.PadSequences(sequences, maxLen);

            // Create a Sequential model
            var model = new Sequential();

            // Add layers to the model
            model.Add(new EmbeddingCuda(vocab.Count, 50));
            model.Add(new GRUCuda(50));
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

            var lrScheduler = new LearningRateScheduler(0.001f, 0.96f, 1000);

            using (var context = new CudaContext())
            {
                var xTrainDevice = new CudaDeviceVariable<float>(xTrain.Length);
                var yTrainDevice = new CudaDeviceVariable<float>(yTrain.Length);

                xTrainDevice.CopyToDevice(xTrain);
                yTrainDevice.CopyToDevice(yTrain);

                // Train the model
                //var stopwatch = new Stopwatch();
                //stopwatch.Start();

                int batchSize = 64;
                int epochs = 50;
                model.Fit(xTrain, yTrain, epochs, batchSize);

                //for (int epoch = 0; epoch < 200; epoch++)
                //{
                //    optimizer.learningRate = lrScheduler.GetLearningRate(epoch);
                //    float loss = model.TrainOnBatch(xTrain, yTrain);

                //    stopwatch.Stop();
                //    Console.WriteLine($"Epoch {epoch}, Loss: {loss} Training time: {stopwatch.Elapsed}");
                //    stopwatch.Restart();
                //}

                //stopwatch.Stop();
            }
            // Create a TextGenerator instance
            var textGenerator = new TextGenerator(model, reverseVocab, maxLen);

            // Generate text
            string seedText = "This is a hole";
            int numWords = 50;
            string generatedText = textGenerator.GenerateText(seedText, numWords);

            // Print the generated text
            Console.WriteLine($"Generated Text: {generatedText}");
        }

        public static void TryImageTraining()
        {
            var (xTrain, yTrain) = LoadImageData("C:\\Users\\SvenW\\source\\repos\\NET_Keras\\NET_Keras\\imgs\\training\\", "500selection.txt");

            var model = new Sequential();
            model.Add(new Conv2D(32, 3, padding: 1));
            model.Add(new MaxPool2D(2));
            model.Add(new Conv2D(64, 3, padding: 1));
            model.Add(new MaxPool2D(2));
            model.Add(new Dense(128, activation: Activations.ReLU));
            model.Add(new Dense(10, activation: Activations.SoftMax));

            var lossFunction = new CategoricalCrossentropy();
            var optimizer = new Adam(0.001f);

            model.Compile(lossFunction, optimizer);

            model.Build(new int[] { -1, 28, 28, 1 });

            Console.WriteLine("Model built.");
            Console.WriteLine("Training...");

            for (int epoch = 0; epoch < 10; epoch++)
            {
                float loss = model.TrainOnBatch4D(xTrain, yTrain);
                Console.WriteLine($"Epoch {epoch}, Loss: {loss}");
            }
        }

        private static (float[,,,] xTrain, float[,] yTrain) LoadImageData(string imageFolderPath, string csvFilePath)
        {
            var images = new List<float[,,]>();
            var labels = new List<int>();

            using (var reader = new StreamReader(csvFilePath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var fields = line.Split('\t');
                    var id = fields[0];
                    var jsonString = fields[2];

                    // Load the image
                    var imagePath = Path.Combine(imageFolderPath, id + ".png");
                    var image = LoadAndPreprocessImage(imagePath);

                    // Parse the label from the JSON string
                    var label = ParseLabelFromJson(jsonString);

                    images.Add(image);
                    labels.Add(label);
                }
            }

            // Convert lists to arrays
            var xTrain = new float[images.Count, 28, 28, 1];
            for (int i = 0; i < images.Count; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    for (int k = 0; k < 28; k++)
                    {
                        xTrain[i, j, k, 0] = images[i][j, k, 0];
                    }
                }
            }

            var yTrain = OneHotEncodeLabels(labels, 10); // Assuming 10 classes

            return (xTrain, yTrain);
        }

        private static float[,,] LoadAndPreprocessImage(string imagePath)
        {
            using (var bitmap = new Bitmap(imagePath))
            {
                // Resize the image to 28x28
                var resizedBitmap = new Bitmap(bitmap, new Size(28, 28));

                // Convert the image to grayscale and normalize pixel values
                var image = new float[28, 28, 1];
                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                    {
                        var pixel = resizedBitmap.GetPixel(i, j);
                        var gray = (pixel.R + pixel.G + pixel.B) / 3.0f / 255.0f;
                        image[i, j, 0] = gray;
                    }
                }

                return image;
            }
        }

        private static int ParseLabelFromJson(string jsonString)
        {
            // Assuming the JSON string contains a field "label" with the class index
            dynamic json = JsonConvert.DeserializeObject(jsonString);
            return (int)json.label;
        }

        private static float[,] OneHotEncodeLabels(List<int> labels, int numClasses)
        {
            var yTrain = new float[labels.Count, numClasses];
            for (int i = 0; i < labels.Count; i++)
            {
                yTrain[i, labels[i]] = 1.0f;
            }
            return yTrain;
        }
    }

}
