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
using NeuralNetwork.Ext;

namespace NET_Keras
{
    class Program
    {
        static void Main()
        {
            TryTextGeneration();
        }

        public static void TryLoadNumbers()
        {
            var model = new Sequential();
            model.Load("1.xml");

            var result = model.Predict(new float[,] { { 9.0f, 10.0f } });
            Console.WriteLine(result[0, 0]);
        }

        public static void TryNumbers()
        { 
            var model = new Sequential();

            model.Add(new DenseCuda(8, activation: Activations.ReLU)
            {
                InputDim = 4
            });
            model.Add(new DenseCuda(3, activation: Activations.ReLU));
            model.Add(new DenseCuda(1, activation: Activations.Linear));

            var lossFunction = new MeanSquaredError();
            var optimizer = new SGD(0.01f);

            model.Compile(lossFunction, optimizer);

            // Dummy training data
            float[,] xTrain = new float[,]
            {
                { 1.0f, 2.0f, 3.0f, 4.0f },
                { 2.0f, 3.0f, 4.0f, 5.0f },
                { 3.0f, 4.0f, 5.0f, 6.0f },
                { 4.0f, 5.0f, 6.0f, 7.0f },
                { 5.0f, 6.0f, 7.0f, 8.0f },
                { 6.0f, 7.0f, 8.0f, 9.0f },
                { 7.0f, 8.0f, 9.0f, 10.0f },
                { 8.0f, 9.0f, 10.0f, 11.0f }
            };

            float[,] yTrain = new float[,]
            {
                { 10.0f },
                { 14.0f },
                { 18.0f },
                { 22.0f },
                { 16.0f },
                { 30.0f },
                { 34.0f },
                { 38.0f }
            };

            // Build the model with the correct input shape
            model.Build(new int[] { xTrain.GetLength(0), xTrain.GetLength(1) });

            model.Fit(xTrain, yTrain, 5, 8, 1);

            // Predict
            float[,] xTest = new float[,] { { 12.0f, 13.0f, 14.0f, 15.0f } };
            float[,] yPred = model.Predict(xTest); // Expected output: 54.0

            // Print the prediction
            Console.WriteLine($"Prediction: {yPred[0, 0]}");

            Console.WriteLine("Do you want to save this model? (Y/N):");
            bool saveModel = Console.ReadLine().ToLower() == "y";

            if (saveModel)
            {
                Console.Write("Enter the model name:");

                string modelName = Console.ReadLine();

                model.Save(modelName);
            }
        }

        public static void TryTextGeneration()
        {
            string text = File.ReadAllText("training_text.txt");

            var vocab = TextPreProcessing.Tokenize(text);
            var reverseVocab = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);

            var sequences = TextPreProcessing.TextToSequences(text, vocab);

            int maxLen = 50;
            sequences = TextPreProcessing.PadSequences(sequences, maxLen);

            var model = new Sequential();

            model.Add(new EmbeddingCuda(vocab.Count, 30));
            model.Add(new LSTMCuda(30));
            model.Add(new DenseCuda(30, activation: ActivationsCuda.ReLU));
            model.Add(new LSTMCuda(15));
            model.Add(new DenseCuda(vocab.Count, activation: ActivationsCuda.SoftMax)); // Output layer

            Console.WriteLine("Layers added.");

            var lossFunction = new CategoricalCrossentropy();
            var optimizer = new Adam(0.01f);

            model.Compile(lossFunction, optimizer);

            Console.WriteLine("Model compiled");

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

            //var lrScheduler = new LearningRateScheduler(0.001f, 0.96f, 1000);

            using (var context = new CudaContext())
            {
                int batchSize = 128;
                int epochs = 10;
                model.Fit(xTrain, yTrain, epochs, batchSize, 1);
            }

            var textGenerator = new TextGenerator(model, reverseVocab, maxLen);

            string seedText = "This is a hole";
            int numWords = 50;
            string generatedText = textGenerator.GenerateText(seedText, numWords);

            Console.WriteLine($"Generated Text: {generatedText}");

            Console.ReadLine();
        }

        public static void TryImageTraining()
        {
            var (xTrain, yTrain) = LoadImageData("\\imgs\\training\\", "500selection.txt");

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
                    var fields = line.Split("__");
                    var id = fields[0];
                    var jsonString = fields[2];

                    if (string.IsNullOrWhiteSpace(jsonString))
                        continue;

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
