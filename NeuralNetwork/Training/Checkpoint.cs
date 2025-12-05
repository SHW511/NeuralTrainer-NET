using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text.Json;
using NeuralNetwork.Models;

namespace NeuralNetwork.Training
{
    /// <summary>
    /// Checkpoint manager for saving and loading Transformer model state.
    /// Saves model weights, optimizer state, and training metadata.
    /// </summary>
    public class Checkpoint
    {
        /// <summary>
        /// Metadata stored with each checkpoint.
        /// </summary>
        public class CheckpointMetadata
        {
            public int Epoch { get; set; }
            public int GlobalStep { get; set; }
            public float TrainLoss { get; set; }
            public float? ValidationLoss { get; set; }
            public float LearningRate { get; set; }
            public DateTime Timestamp { get; set; }
            public string ModelConfigJson { get; set; }
            public Dictionary<string, string> CustomData { get; set; }
        }

        /// <summary>
        /// Save a TransformerLM checkpoint.
        /// </summary>
        /// <param name="path">Path to save checkpoint (without extension).</param>
        /// <param name="model">The model to save.</param>
        /// <param name="config">Model configuration.</param>
        /// <param name="epoch">Current epoch.</param>
        /// <param name="globalStep">Current global training step.</param>
        /// <param name="trainLoss">Current training loss.</param>
        /// <param name="validationLoss">Current validation loss (optional).</param>
        /// <param name="learningRate">Current learning rate.</param>
        /// <param name="optimizerState">Optional optimizer state to save.</param>
        public static void Save(
            string path,
            TransformerLM model,
            TransformerConfig config,
            int epoch,
            int globalStep,
            float trainLoss,
            float? validationLoss = null,
            float learningRate = 0f,
            Dictionary<string, float[,]> optimizerState = null)
        {
            // Create checkpoint directory
            string checkpointDir = path;
            if (!Directory.Exists(checkpointDir))
                Directory.CreateDirectory(checkpointDir);

            // Save metadata
            var metadata = new CheckpointMetadata
            {
                Epoch = epoch,
                GlobalStep = globalStep,
                TrainLoss = trainLoss,
                ValidationLoss = validationLoss,
                LearningRate = learningRate,
                Timestamp = DateTime.UtcNow,
                ModelConfigJson = JsonSerializer.Serialize(config),
                CustomData = new Dictionary<string, string>()
            };

            string metadataPath = Path.Combine(checkpointDir, "metadata.json");
            File.WriteAllText(metadataPath, JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));

            // Save model config separately for easy inspection
            string configPath = Path.Combine(checkpointDir, "config.json");
            config.Save(configPath);

            // Save model weights
            var parameters = model.GetParameters();
            string weightsDir = Path.Combine(checkpointDir, "weights");
            if (!Directory.Exists(weightsDir))
                Directory.CreateDirectory(weightsDir);

            foreach (var (name, weights, _) in parameters)
            {
                SaveArray(Path.Combine(weightsDir, $"{SanitizeFileName(name)}.bin"), weights);
            }

            // Save optimizer state if provided
            if (optimizerState != null && optimizerState.Count > 0)
            {
                string optimizerDir = Path.Combine(checkpointDir, "optimizer");
                if (!Directory.Exists(optimizerDir))
                    Directory.CreateDirectory(optimizerDir);

                foreach (var (name, state) in optimizerState)
                {
                    SaveArray(Path.Combine(optimizerDir, $"{SanitizeFileName(name)}.bin"), state);
                }
            }

            Console.WriteLine($"Checkpoint saved to {checkpointDir}");
        }

        /// <summary>
        /// Load a TransformerLM checkpoint.
        /// </summary>
        /// <param name="path">Path to checkpoint directory.</param>
        /// <param name="model">Model to load weights into (must match architecture).</param>
        /// <returns>Checkpoint metadata.</returns>
        public static CheckpointMetadata Load(string path, TransformerLM model)
        {
            if (!Directory.Exists(path))
                throw new DirectoryNotFoundException($"Checkpoint not found: {path}");

            // Load metadata
            string metadataPath = Path.Combine(path, "metadata.json");
            var metadata = JsonSerializer.Deserialize<CheckpointMetadata>(File.ReadAllText(metadataPath));

            // Load model weights
            var parameters = model.GetParameters();
            string weightsDir = Path.Combine(path, "weights");

            foreach (var (name, weights, _) in parameters)
            {
                string weightPath = Path.Combine(weightsDir, $"{SanitizeFileName(name)}.bin");
                if (File.Exists(weightPath))
                {
                    LoadArray(weightPath, weights);
                }
                else
                {
                    Console.WriteLine($"Warning: Weight file not found for {name}");
                }
            }

            Console.WriteLine($"Checkpoint loaded from {path} (epoch {metadata.Epoch}, step {metadata.GlobalStep})");
            return metadata;
        }

        /// <summary>
        /// Load optimizer state from checkpoint.
        /// </summary>
        /// <param name="path">Path to checkpoint directory.</param>
        /// <param name="optimizerState">Dictionary to load state into.</param>
        public static void LoadOptimizerState(string path, Dictionary<string, float[,]> optimizerState)
        {
            string optimizerDir = Path.Combine(path, "optimizer");
            if (!Directory.Exists(optimizerDir))
                return;

            foreach (var file in Directory.GetFiles(optimizerDir, "*.bin"))
            {
                string name = Path.GetFileNameWithoutExtension(file);
                if (optimizerState.TryGetValue(name, out var state))
                {
                    LoadArray(file, state);
                }
            }
        }

        /// <summary>
        /// Load just the configuration from a checkpoint.
        /// </summary>
        public static TransformerConfig LoadConfig(string path)
        {
            string configPath = Path.Combine(path, "config.json");
            return TransformerConfig.Load(configPath);
        }

        /// <summary>
        /// Get metadata from a checkpoint without loading weights.
        /// </summary>
        public static CheckpointMetadata GetMetadata(string path)
        {
            string metadataPath = Path.Combine(path, "metadata.json");
            return JsonSerializer.Deserialize<CheckpointMetadata>(File.ReadAllText(metadataPath));
        }

        /// <summary>
        /// List all checkpoints in a directory.
        /// </summary>
        public static List<(string path, CheckpointMetadata metadata)> ListCheckpoints(string directory)
        {
            var checkpoints = new List<(string, CheckpointMetadata)>();

            if (!Directory.Exists(directory))
                return checkpoints;

            foreach (var subdir in Directory.GetDirectories(directory))
            {
                string metadataPath = Path.Combine(subdir, "metadata.json");
                if (File.Exists(metadataPath))
                {
                    try
                    {
                        var metadata = GetMetadata(subdir);
                        checkpoints.Add((subdir, metadata));
                    }
                    catch
                    {
                        // Skip invalid checkpoints
                    }
                }
            }

            return checkpoints;
        }

        #region Binary I/O Helpers

        private static void SaveArray(string path, float[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);

            using var stream = new FileStream(path, FileMode.Create);
            using var writer = new BinaryWriter(stream);

            // Write dimensions
            writer.Write(rows);
            writer.Write(cols);

            // Write data
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    writer.Write(array[i, j]);
                }
            }
        }

        private static void LoadArray(string path, float[,] array)
        {
            using var stream = new FileStream(path, FileMode.Open);
            using var reader = new BinaryReader(stream);

            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();

            if (rows != array.GetLength(0) || cols != array.GetLength(1))
            {
                throw new InvalidDataException(
                    $"Array dimension mismatch: expected [{array.GetLength(0)}, {array.GetLength(1)}], " +
                    $"got [{rows}, {cols}]");
            }

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    array[i, j] = reader.ReadSingle();
                }
            }
        }

        private static string SanitizeFileName(string name)
        {
            // Replace invalid filename characters
            foreach (char c in Path.GetInvalidFileNameChars())
            {
                name = name.Replace(c, '_');
            }
            return name;
        }

        #endregion
    }

    /// <summary>
    /// Manages automatic checkpointing during training.
    /// </summary>
    public class CheckpointManager
    {
        private readonly string _checkpointDir;
        private readonly int _maxCheckpoints;
        private readonly List<string> _checkpointPaths;

        public string CheckpointDirectory => _checkpointDir;

        /// <summary>
        /// Create a checkpoint manager.
        /// </summary>
        /// <param name="checkpointDir">Directory to store checkpoints.</param>
        /// <param name="maxCheckpoints">Maximum number of checkpoints to keep (0 = unlimited).</param>
        public CheckpointManager(string checkpointDir, int maxCheckpoints = 5)
        {
            _checkpointDir = checkpointDir;
            _maxCheckpoints = maxCheckpoints;
            _checkpointPaths = new List<string>();

            if (!Directory.Exists(_checkpointDir))
                Directory.CreateDirectory(_checkpointDir);

            // Load existing checkpoint list
            foreach (var (path, _) in Checkpoint.ListCheckpoints(_checkpointDir))
            {
                _checkpointPaths.Add(path);
            }
        }

        /// <summary>
        /// Save a new checkpoint and manage old ones.
        /// </summary>
        public string SaveCheckpoint(
            TransformerLM model,
            TransformerConfig config,
            int epoch,
            int globalStep,
            float trainLoss,
            float? validationLoss = null,
            float learningRate = 0f)
        {
            // Create checkpoint name
            string checkpointName = $"checkpoint_epoch{epoch}_step{globalStep}";
            string checkpointPath = Path.Combine(_checkpointDir, checkpointName);

            // Save checkpoint
            Checkpoint.Save(checkpointPath, model, config, epoch, globalStep, trainLoss, validationLoss, learningRate);

            _checkpointPaths.Add(checkpointPath);

            // Remove old checkpoints if exceeding limit
            while (_maxCheckpoints > 0 && _checkpointPaths.Count > _maxCheckpoints)
            {
                string oldCheckpoint = _checkpointPaths[0];
                _checkpointPaths.RemoveAt(0);

                try
                {
                    Directory.Delete(oldCheckpoint, recursive: true);
                    Console.WriteLine($"Removed old checkpoint: {oldCheckpoint}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Could not delete checkpoint {oldCheckpoint}: {ex.Message}");
                }
            }

            return checkpointPath;
        }

        /// <summary>
        /// Get the path to the latest checkpoint.
        /// </summary>
        public string GetLatestCheckpoint()
        {
            if (_checkpointPaths.Count == 0)
                return null;

            return _checkpointPaths[_checkpointPaths.Count - 1];
        }

        /// <summary>
        /// Load the latest checkpoint into a model.
        /// </summary>
        public Checkpoint.CheckpointMetadata LoadLatest(TransformerLM model)
        {
            string latest = GetLatestCheckpoint();
            if (latest == null)
                throw new InvalidOperationException("No checkpoints available");

            return Checkpoint.Load(latest, model);
        }
    }
}
