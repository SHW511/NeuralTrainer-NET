using System.Diagnostics;
using System.Drawing;
using ManagedCuda;
using NeuralNetwork;
using NeuralNetwork.Ext;
using NeuralNetwork.Inference;
using NeuralNetwork.Layers;
using NeuralNetwork.Layers.Activations;
using NeuralNetwork.Layers.Cuda;
using NeuralNetwork.Losses;
using NeuralNetwork.Models;
using NeuralNetwork.Optimizers;
using NeuralNetwork.Processing.Text;
using NeuralNetwork.Tensors;
using NeuralNetwork.Training;
using Newtonsoft.Json;

namespace NET_Keras
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("=== NeuralTrainer-NET ===\n");
            Console.WriteLine("Select compute device:");
            Console.WriteLine("  1. CPU (default)");
            Console.WriteLine("  2. GPU (CUDA)");
            Console.Write("\nChoice [1]: ");

            string input = Console.ReadLine()?.Trim();
            bool useCuda = input == "2";

            if (useCuda)
            {
                Console.WriteLine("\nUsing GPU (CUDA) acceleration");
            }
            else
            {
                Console.WriteLine("\nUsing CPU");
            }

            Console.WriteLine("\nSelect training mode:");
            Console.WriteLine("  1. Quick Test (tiny model, ~10 sec)");
            Console.WriteLine("  2. Transformer Text Generation");
            Console.WriteLine("  3. Transformer Character-Level");
            Console.WriteLine("  4. Transformer Full Training");
            Console.Write("\nChoice [1]: ");

            string modeInput = Console.ReadLine()?.Trim();

            Console.WriteLine();

            switch (modeInput)
            {
                case "2":
                    TryTransformerTextGeneration(useCuda);
                    break;
                case "3":
                    TryTransformerCharLevel(useCuda);
                    break;
                case "4":
                    TryTransformerFullTraining(useCuda);
                    break;
                default:
                    TryTransformerQuickTest(useCuda);
                    break;
            }
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

            model.Add(new DenseCuda(8, activation: Activations.Linear)
            {
                InputDim = 4
            });
            model.Add(new DenseCuda(4, activation: Activations.Linear));
            model.Add(new DenseCuda(2, activation: Activations.Linear));
            model.Add(new DenseCuda(1, activation: Activations.Linear));

            var lossFunction = new MeanSquaredError();
            var optimizer = new Adam();

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

            model.Fit(xTrain, yTrain, 5, 32, 1);

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

            model.Add(new DenseCuda(maxLen, activation: Activations.Linear)
            {
                InputDim = vocab.Count
            });
            model.Add(new EmbeddingCuda(vocab.Count, 30));
            model.Add(new LSTMCuda(30));
            //model.Add(new DenseCuda(30, activation: ActivationsCuda.ReLU));
            //model.Add(new LSTMCuda(15));
            model.Add(new DenseCuda(vocab.Count, activation: ActivationsCuda.SoftMax)); // Output layer

            Console.WriteLine("Layers added.");

            var lossFunction = new CategoricalCrossentropy();
            var optimizer = new Adam(0.0001f);

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
                int batchSize = 64;
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

        /// <summary>
        /// Quick test with minimal model for fast verification.
        /// Uses character-level tokenization and tiny model (1 layer, 32 dim).
        /// </summary>
        /// <param name="useCuda">If true, use CUDA GPU acceleration.</param>
        public static void TryTransformerQuickTest(bool useCuda = false)
        {
            Console.WriteLine($"=== Quick Test ({(useCuda ? "CUDA" : "CPU")}) ===\n");
            var stopwatch = new Stopwatch();
            stopwatch.Start();

            // Load text
            string text = File.ReadAllText("training_text.txt");
            Console.WriteLine($"Loaded {text.Length:N0} characters");

            // Character-level tokenizer (instant, no training needed)
            var tokenizer = BPETokenizer.CreateCharacterLevel(text);
            Console.WriteLine($"Vocabulary size: {tokenizer.VocabSize}");

            // Minimal settings
            int seqLength = 32;
            int batchSize = 4;
            var dataset = TextDataset.FromText(text, tokenizer, seqLength, batchSize, shuffle: true, seed: 42);
            Console.WriteLine($"Batches: {dataset.NumBatches}");

            // Tiny model config
            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSeqLen = seqLength,
                NumLayers = 1,              // Single layer
                NumHeads = 2,               // 2 attention heads
                EmbeddingDim = 32,          // Tiny embedding
                FFNDim = 64,                // Small FFN
                DropoutRate = 0f,           // No dropout for fast test
                LearningRate = 1e-3f
            };
            config.Validate();

            // Create model
            dynamic model;
            IDisposable disposableModel = null;

            if (useCuda)
            {
                var cudaModel = new TransformerLMCuda(config);
                model = cudaModel;
                disposableModel = cudaModel;
            }
            else
            {
                model = new TransformerLM(config);
            }

            try
            {
                Console.WriteLine($"Parameters: {model.CountParameters():N0}");
                Console.WriteLine("\nTraining (1 epoch)...");

                float totalLoss = 0f;
                int batchCount = 0;
                int maxBatches = Math.Min(20, dataset.NumBatches); // Limit batches for speed

                foreach (var (inputs, targets) in dataset.GetBatches())
                {
                    if (batchCount >= maxBatches) break;

                    var logits = model.Forward(inputs);
                    float loss = model.ComputeLoss(logits, targets);
                    totalLoss += loss;
                    batchCount++;

                    model.Backward(logits, targets);

                    // SGD update
                    List<(string name, float[,] weights, float[,] gradients)> parameters = model.GetParameters();
                    foreach (var (_, weights, grads) in parameters)
                    {
                        for (int i = 0; i < weights.GetLength(0); i++)
                            for (int j = 0; j < weights.GetLength(1); j++)
                                weights[i, j] -= config.LearningRate * Math.Clamp(grads[i, j], -1f, 1f);
                    }

                    if (useCuda) model.SyncToDevice();

                    Console.Write($"\rBatch {batchCount}/{maxBatches} - Loss: {totalLoss / batchCount:F4}");
                }

                stopwatch.Stop();
                Console.WriteLine($"\n\nTraining complete in {stopwatch.Elapsed.TotalSeconds:F1}s");
                Console.WriteLine($"Final loss: {totalLoss / batchCount:F4}");

                // Quick generation test
                Console.WriteLine("\n--- Generation Test ---");
                model.Training = false;

                string seedText = "The";
                int[] seedTokens = tokenizer.Encode(seedText);
                int[,] tokens = new int[1, seqLength];

                for (int i = 0; i < seqLength; i++)
                    tokens[0, i] = tokenizer.PadId;

                int startPos = seqLength - seedTokens.Length;
                for (int i = 0; i < seedTokens.Length; i++)
                    tokens[0, startPos + i] = seedTokens[i];

                var generated = new List<int>();
                for (int i = 0; i < 20; i++)
                {
                    var logits = model.Forward(tokens);

                    // Greedy sampling
                    int bestToken = 0;
                    float bestLogit = float.NegativeInfinity;
                    for (int v = 0; v < config.VocabSize; v++)
                    {
                        if (v == tokenizer.PadId) continue;
                        if (logits[0, seqLength - 1, v] > bestLogit)
                        {
                            bestLogit = logits[0, seqLength - 1, v];
                            bestToken = v;
                        }
                    }

                    generated.Add(bestToken);
                    if (bestToken == tokenizer.EosId) break;

                    for (int t = 0; t < seqLength - 1; t++)
                        tokens[0, t] = tokens[0, t + 1];
                    tokens[0, seqLength - 1] = bestToken;
                }

                Console.WriteLine($"Seed: \"{seedText}\"");
                Console.WriteLine($"Generated: {seedText}{tokenizer.Decode(generated.ToArray())}");

                Console.WriteLine("\n=== Quick Test Complete ===");
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey();
            }
            finally
            {
                disposableModel?.Dispose();
            }
        }

        /// <summary>
        /// Transformer-based text generation training example.
        /// Uses the new Transformer architecture with BPE tokenization.
        /// </summary>
        /// <param name="useCuda">If true, use CUDA GPU acceleration.</param>
        public static void TryTransformerTextGeneration(bool useCuda = false)
        {
            Console.WriteLine($"=== Transformer Text Generation Training ({(useCuda ? "CUDA" : "CPU")}) ===\n");

            // ===== 1. Load Training Text =====
            Console.WriteLine("Loading training text...");
            string text = File.ReadAllText("training_text.txt");
            Console.WriteLine($"Loaded {text.Length:N0} characters");

            // ===== 2. Train BPE Tokenizer =====
            Console.WriteLine("\n--- Training BPE Tokenizer ---");
            var tokenizer = new BPETokenizer();

            // Target vocabulary size (smaller for demo, typically 32k-50k)
            int targetVocabSize = 1000;
            tokenizer.Train(text, targetVocabSize, minFrequency: 2);

            Console.WriteLine(tokenizer.GetStats());

            // Save tokenizer for later use
            string tokenizerPath = "tokenizer";
            tokenizer.Save(tokenizerPath);

            // ===== 3. Create Dataset =====
            Console.WriteLine("\n--- Creating Dataset ---");
            int seqLength = 64;
            int batchSize = 8;

            var dataset = TextDataset.FromText(text, tokenizer, seqLength, batchSize, shuffle: true, seed: 42);
            Console.WriteLine(dataset.GetStats());

            // Optionally split into train/val
            // var (trainData, valData) = dataset.Split(0.1f);

            // ===== 4. Configure the Transformer Model =====
            Console.WriteLine("\n--- Configuring Transformer Model ---");

            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSeqLen = seqLength,
                NumLayers = 4,              // 4 Transformer blocks
                NumHeads = 4,               // 4 attention heads
                EmbeddingDim = 128,         // Model dimension
                FFNDim = 512,               // Feed-forward dimension (4x embedding)
                DropoutRate = 0.1f,
                LearningRate = 1e-4f,
                WarmupSteps = 1000,
                TieEmbeddings = true,       // Share embedding and output weights
                UseCausalMask = true        // Autoregressive generation
            };

            config.Validate();
            Console.WriteLine(config);
            Console.WriteLine($"Head dimension: {config.HeadDim}");

            // ===== 5. Create the Model (CPU or CUDA) =====
            Console.WriteLine($"\nCreating Transformer Language Model ({(useCuda ? "CUDA" : "CPU")})...");

            // Use dynamic to handle both model types with same interface
            dynamic model;
            IDisposable disposableModel = null;

            if (useCuda)
            {
                var cudaModel = new TransformerLMCuda(config);
                model = cudaModel;
                disposableModel = cudaModel;
            }
            else
            {
                model = new TransformerLM(config);
            }

            try
            {
                long numParams = model.CountParameters();
                Console.WriteLine($"Total trainable parameters: {numParams:N0}");

                // ===== 6. Training Loop =====
                Console.WriteLine("\n--- Starting Training ---");

                int epochs = 3;
                var stopwatch = new Stopwatch();
                float learningRate = config.LearningRate;

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    stopwatch.Restart();
                    float epochLoss = 0f;
                    int batchCount = 0;

                    // Reset dataset for new epoch
                    dataset.Reset();

                    while (dataset.HasNextBatch())
                    {
                        // Get batch from dataset
                        var (inputTokens, targetTokens) = dataset.GetNextBatch();

                        // Forward pass
                        Tensor logits = model.Forward(inputTokens);

                        // Compute loss
                        float loss = model.ComputeLoss(logits, targetTokens);
                        epochLoss += loss;
                        batchCount++;

                        // Backward pass
                        model.Backward(logits, targetTokens);

                        // Simple gradient descent update
                        List<(string name, float[,] weights, float[,] gradients)> parameters = model.GetParameters();
                        foreach (var (name, weights, gradients) in parameters)
                        {
                            for (int i = 0; i < weights.GetLength(0); i++)
                            {
                                for (int j = 0; j < weights.GetLength(1); j++)
                                {
                                    // Gradient clipping
                                    float grad = gradients[i, j];
                                    grad = Math.Clamp(grad, -config.GradientClip, config.GradientClip);

                                    // Update weight
                                    weights[i, j] -= learningRate * grad;
                                }
                            }
                        }

                        // Sync to device after weight updates (CUDA only)
                        if (useCuda)
                        {
                            model.SyncToDevice();
                        }

                        // Progress reporting
                        if (batchCount % 10 == 0 || !dataset.HasNextBatch())
                        {
                            Console.Write($"\rEpoch {epoch + 1}/{epochs} - Batch {batchCount}/{dataset.NumBatches} - Loss: {epochLoss / batchCount:F4}");
                        }
                    }

                    stopwatch.Stop();
                    float avgLoss = epochLoss / batchCount;
                    Console.WriteLine($"\rEpoch {epoch + 1}/{epochs} completed - Avg Loss: {avgLoss:F4} - Time: {stopwatch.Elapsed.TotalSeconds:F1}s          ");
                }

                // ===== 7. Text Generation Example =====
                Console.WriteLine("\n=== Text Generation ===");

                model.Training = false;  // Switch to inference mode (disable dropout)

                // Generate text from a seed
                string seedText = "This is";
                Console.WriteLine($"Seed: \"{seedText}\"");
                Console.Write("Generated: ");

                // Tokenize seed using BPE
                int[] seedTokens = tokenizer.Encode(seedText);
                int[,] tokens = new int[1, seqLength];

                // Initialize with PAD tokens
                for (int i = 0; i < seqLength; i++)
                    tokens[0, i] = tokenizer.PadId;

                // Place seed tokens at the end of the sequence
                int startPos = Math.Max(0, seqLength - seedTokens.Length);
                for (int i = 0; i < seedTokens.Length && startPos + i < seqLength; i++)
                {
                    tokens[0, startPos + i] = seedTokens[i];
                }

                // Generate tokens autoregressively
                int tokensToGenerate = 30;
                var generatedIds = new List<int>();

                for (int i = 0; i < tokensToGenerate; i++)
                {
                    // Forward pass
                    Tensor logits = model.Forward(tokens);

                    // Get logits for the last position
                    int lastPos = seqLength - 1;
                    int vocabSize = config.VocabSize;

                    // Simple argmax sampling (greedy)
                    int bestToken = 0;
                    float bestLogit = float.NegativeInfinity;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        // Skip special tokens during generation
                        if (v == tokenizer.PadId || v == tokenizer.UnkId)
                            continue;

                        if (logits[0, lastPos, v] > bestLogit)
                        {
                            bestLogit = logits[0, lastPos, v];
                            bestToken = v;
                        }
                    }

                    generatedIds.Add(bestToken);

                    // Stop if EOS is generated
                    if (bestToken == tokenizer.EosId)
                        break;

                    // Shift tokens left and add the new token
                    for (int t = 0; t < seqLength - 1; t++)
                    {
                        tokens[0, t] = tokens[0, t + 1];
                    }
                    tokens[0, seqLength - 1] = bestToken;
                }

                // Decode and print generated text
                string generatedText = tokenizer.Decode(generatedIds.ToArray());
                Console.WriteLine(seedText + generatedText);

                Console.WriteLine("\n=== Training Complete ===");
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey();
            }
            finally
            {
                // Dispose CUDA resources if applicable
                disposableModel?.Dispose();
            }
        }

        /// <summary>
        /// Alternative training method using character-level tokenization (simpler, for quick testing).
        /// </summary>
        /// <param name="useCuda">If true, use CUDA GPU acceleration.</param>
        public static void TryTransformerCharLevel(bool useCuda = false)
        {
            Console.WriteLine($"=== Transformer Character-Level Training ({(useCuda ? "CUDA" : "CPU")}) ===\n");

            // Load text
            string text = File.ReadAllText("training_text.txt");
            Console.WriteLine($"Loaded {text.Length:N0} characters");

            // Create character-level tokenizer
            var tokenizer = BPETokenizer.CreateCharacterLevel(text);
            Console.WriteLine($"Vocabulary size: {tokenizer.VocabSize}");

            // Create dataset
            int seqLength = 128;
            int batchSize = 16;
            var dataset = TextDataset.FromText(text, tokenizer, seqLength, batchSize);
            Console.WriteLine(dataset.GetStats());

            // Configure smaller model for char-level
            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSeqLen = seqLength,
                NumLayers = 2,
                NumHeads = 4,
                EmbeddingDim = 64,
                FFNDim = 256,
                DropoutRate = 0.1f,
                LearningRate = 5e-4f
            };

            // Create model (CPU or CUDA)
            dynamic model;
            IDisposable disposableModel = null;

            if (useCuda)
            {
                var cudaModel = new TransformerLMCuda(config);
                model = cudaModel;
                disposableModel = cudaModel;
            }
            else
            {
                model = new TransformerLM(config);
            }

            try
            {
                Console.WriteLine($"Parameters: {model.CountParameters():N0}");

                // Quick training (1 epoch)
                float totalLoss = 0f;
                int batchCount = 0;

                foreach (var (inputs, targets) in dataset.GetBatches())
                {
                    var logits = model.Forward(inputs);
                    float loss = model.ComputeLoss(logits, targets);
                    totalLoss += loss;
                    batchCount++;

                    model.Backward(logits, targets);

                    // SGD update
                    List<(string name, float[,] weights, float[,] gradients)> parameters = model.GetParameters();
                    foreach (var (_, weights, grads) in parameters)
                    {
                        for (int i = 0; i < weights.GetLength(0); i++)
                            for (int j = 0; j < weights.GetLength(1); j++)
                                weights[i, j] -= config.LearningRate * Math.Clamp(grads[i, j], -1f, 1f);
                    }

                    // Sync to device after weight updates (CUDA only)
                    if (useCuda)
                    {
                        model.SyncToDevice();
                    }

                    if (batchCount % 20 == 0)
                        Console.Write($"\rBatch {batchCount}/{dataset.NumBatches} - Loss: {totalLoss / batchCount:F4}");
                }

                Console.WriteLine($"\nFinal loss: {totalLoss / batchCount:F4}");
            }
            finally
            {
                disposableModel?.Dispose();
            }
        }

        /// <summary>
        /// Full-featured Transformer training with LR scheduling, gradient accumulation,
        /// checkpointing, and advanced sampling.
        /// </summary>
        /// <param name="useCuda">If true, use CUDA GPU acceleration.</param>
        public static void TryTransformerFullTraining(bool useCuda = false)
        {
            Console.WriteLine($"=== Full-Featured Transformer Training ({(useCuda ? "CUDA" : "CPU")}) ===\n");

            // ===== 1. Configuration =====
            string dataPath = "training_text.txt";
            string checkpointDir = "checkpoints";
            string tokenizerDir = "tokenizer";

            int vocabSize = 2000;
            int seqLength = 128;
            int batchSize = 4;
            int accumulationSteps = 4;  // Effective batch size = 4 * 4 = 16
            int epochs = 5;
            int checkpointEveryNSteps = 500;

            // ===== 2. Load Data and Train Tokenizer =====
            Console.WriteLine("Loading data...");
            string text = File.ReadAllText(dataPath);
            Console.WriteLine($"Loaded {text.Length:N0} characters");

            BPETokenizer tokenizer;
            if (Directory.Exists(tokenizerDir))
            {
                Console.WriteLine("Loading existing tokenizer...");
                tokenizer = BPETokenizer.Load(tokenizerDir);
            }
            else
            {
                Console.WriteLine("Training new tokenizer...");
                tokenizer = new BPETokenizer();
                tokenizer.Train(text, vocabSize);
                tokenizer.Save(tokenizerDir);
            }
            Console.WriteLine(tokenizer.GetStats());

            // ===== 3. Create Dataset =====
            Console.WriteLine("\nCreating dataset...");
            var fullDataset = TextDataset.FromText(text, tokenizer, seqLength, batchSize, shuffle: true, seed: 42);
            var (trainData, valData) = fullDataset.Split(0.1f);
            Console.WriteLine($"Train: {trainData.GetStats()}");
            Console.WriteLine($"Val: {valData.GetStats()}");

            // ===== 4. Configure Model =====
            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                MaxSeqLen = seqLength,
                NumLayers = 4,
                NumHeads = 4,
                EmbeddingDim = 256,
                FFNDim = 1024,
                DropoutRate = 0.1f,
                LearningRate = 3e-4f,
                WarmupSteps = 500,
                GradientClip = 1.0f,
                TieEmbeddings = true,
                UseCausalMask = true
            };
            config.Validate();
            Console.WriteLine($"\nModel: {config}");

            // ===== 5. Create Model (CPU or CUDA) =====
            dynamic model;
            IDisposable disposableModel = null;

            if (useCuda)
            {
                var cudaModel = new TransformerLMCuda(config);
                model = cudaModel;
                disposableModel = cudaModel;
            }
            else
            {
                model = new TransformerLM(config);
            }

            try
            {
                Console.WriteLine($"Parameters: {model.CountParameters():N0}");

                // ===== 6. Create Training Components =====
                int totalSteps = trainData.NumBatches * epochs / accumulationSteps;
                var lrScheduler = NeuralNetwork.Training.LearningRateScheduler.CreateCosineScheduler(
                    peakLr: config.LearningRate,
                    warmupSteps: config.WarmupSteps,
                    totalSteps: totalSteps,
                    minLr: 1e-6f);

                var gradAccumulator = new GradientAccumulator(accumulationSteps);
                var checkpointManager = new CheckpointManager(checkpointDir, maxCheckpoints: 3);

                // Try to resume from checkpoint (only for CPU model, CUDA checkpoint loading would need additional work)
                int startEpoch = 0;
                int globalStep = 0;

                if (!useCuda)
                {
                    string latestCheckpoint = checkpointManager.GetLatestCheckpoint();
                    if (latestCheckpoint != null)
                    {
                        Console.WriteLine($"\nResuming from checkpoint: {latestCheckpoint}");
                        var metadata = Checkpoint.Load(latestCheckpoint, model);
                        startEpoch = metadata.Epoch;
                        globalStep = metadata.GlobalStep;
                        lrScheduler.SetStep(globalStep);
                        Console.WriteLine($"Resumed at epoch {startEpoch}, step {globalStep}");
                    }
                }

                // ===== 7. Training Loop =====
                Console.WriteLine("\n--- Starting Training ---");
                var stopwatch = new Stopwatch();
                float bestValLoss = float.MaxValue;

                for (int epoch = startEpoch; epoch < epochs; epoch++)
                {
                    stopwatch.Restart();
                    model.Training = true;

                    float epochLoss = 0f;
                    int microBatchCount = 0;
                    int updateCount = 0;

                    trainData.Reset();

                    while (trainData.HasNextBatch())
                    {
                        var (inputs, targets) = trainData.GetNextBatch();

                        // Forward pass
                        var logits = model.Forward(inputs);
                        float loss = model.ComputeLoss(logits, targets);
                        epochLoss += loss;
                        microBatchCount++;

                        // Backward pass
                        model.Backward(logits, targets);

                        // Accumulate gradients
                        List<(string name, float[,] weights, float[,] gradients)> parameters = model.GetParameters();
                        gradAccumulator.Accumulate(parameters);

                        // Update weights when accumulation is complete
                        if (gradAccumulator.ShouldUpdate)
                        {
                            globalStep++;
                            float lr = lrScheduler.Step();

                            // Apply accumulated gradients
                            gradAccumulator.ApplyAccumulatedGradients(parameters);

                            // Gradient clipping
                            float gradNorm = GradientUtils.ClipGradientsByNorm(parameters, config.GradientClip);

                            // SGD update (could be replaced with Adam)
                            foreach (var (name, weights, gradients) in parameters)
                            {
                                for (int i = 0; i < weights.GetLength(0); i++)
                                {
                                    for (int j = 0; j < weights.GetLength(1); j++)
                                    {
                                        weights[i, j] -= lr * gradients[i, j];
                                    }
                                }
                            }

                            // Sync to device after weight updates (CUDA only)
                            if (useCuda)
                            {
                                model.SyncToDevice();
                            }

                            updateCount++;

                            // Checkpoint (only for CPU model for now)
                            if (!useCuda && globalStep % checkpointEveryNSteps == 0)
                            {
                                float avgLoss = epochLoss / microBatchCount;
                                checkpointManager.SaveCheckpoint(model, config, epoch, globalStep, avgLoss, learningRate: lr);
                            }
                        }

                        // Progress
                        if (microBatchCount % 20 == 0)
                        {
                            float avgLoss = epochLoss / microBatchCount;
                            Console.Write($"\rEpoch {epoch + 1}/{epochs} | Batch {microBatchCount}/{trainData.NumBatches} | " +
                                         $"Loss: {avgLoss:F4} | LR: {lrScheduler.CurrentLr:E2}");
                        }
                    }

                    // Validation
                    model.Training = false;
                    float valLoss = 0f;
                    int valBatches = 0;
                    valData.Reset();

                    while (valData.HasNextBatch())
                    {
                        var (inputs, targets) = valData.GetNextBatch();
                        var logits = model.Forward(inputs);
                        valLoss += model.ComputeLoss(logits, targets);
                        valBatches++;
                    }

                    float avgValLoss = valLoss / valBatches;
                    float avgTrainLoss = epochLoss / microBatchCount;

                    stopwatch.Stop();
                    Console.WriteLine($"\rEpoch {epoch + 1}/{epochs} | Train Loss: {avgTrainLoss:F4} | " +
                                     $"Val Loss: {avgValLoss:F4} | Time: {stopwatch.Elapsed.TotalSeconds:F1}s          ");

                    // Save best model (only for CPU model for now)
                    if (!useCuda && avgValLoss < bestValLoss)
                    {
                        bestValLoss = avgValLoss;
                        checkpointManager.SaveCheckpoint(model, config, epoch, globalStep, avgTrainLoss, avgValLoss, lrScheduler.CurrentLr);
                        Console.WriteLine($"  New best model saved (val_loss: {avgValLoss:F4})");
                    }
                }

                // ===== 8. Text Generation Demo =====
                Console.WriteLine("\n=== Text Generation with Advanced Sampling ===");
                model.Training = false;

                var sampler = Sampler.Creative(seed: 42);
                Console.WriteLine($"Sampler: {sampler}");

                string seedText = "The";
                Console.WriteLine($"\nSeed: \"{seedText}\"");

                int[] seedTokens = tokenizer.Encode(seedText);
                var generatedTokens = new List<int>(seedTokens);
                int[,] tokens = new int[1, seqLength];

                // Initialize with padding
                for (int i = 0; i < seqLength; i++)
                    tokens[0, i] = tokenizer.PadId;

                // Place seed at end
                int startPos = seqLength - seedTokens.Length;
                for (int i = 0; i < seedTokens.Length; i++)
                    tokens[0, startPos + i] = seedTokens[i];

                Console.Write("Generated: " + seedText);

                int tokensToGenerate = 50;
                for (int i = 0; i < tokensToGenerate; i++)
                {
                    var logits = model.Forward(tokens);

                    // Use sampler with repetition penalty
                    int nextToken = sampler.SampleFromTensor(logits, -1, 0, generatedTokens);
                    generatedTokens.Add(nextToken);

                    if (nextToken == tokenizer.EosId)
                        break;

                    // Shift and add new token
                    for (int t = 0; t < seqLength - 1; t++)
                        tokens[0, t] = tokens[0, t + 1];
                    tokens[0, seqLength - 1] = nextToken;
                }

                string generated = tokenizer.Decode(generatedTokens.Skip(seedTokens.Length).ToArray());
                Console.WriteLine(generated);

                Console.WriteLine("\n=== Training Complete ===");
                Console.ReadKey();
            }
            finally
            {
                disposableModel?.Dispose();
            }
        }
    }

}
