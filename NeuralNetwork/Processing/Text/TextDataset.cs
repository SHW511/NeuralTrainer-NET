using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork.Processing.Text
{
    /// <summary>
    /// Dataset for language model training.
    /// Handles tokenization, batching, and creating input/target pairs for next-token prediction.
    /// </summary>
    public class TextDataset
    {
        private readonly int[] _tokenIds;
        private readonly int _seqLength;
        private readonly int _batchSize;
        private readonly bool _shuffle;
        private readonly Random _random;

        private int[] _indices;
        private int _currentPosition;

        /// <summary>
        /// Total number of tokens in the dataset.
        /// </summary>
        public int TotalTokens => _tokenIds.Length;

        /// <summary>
        /// Number of sequences that can be formed.
        /// </summary>
        public int NumSequences => Math.Max(0, _tokenIds.Length - _seqLength);

        /// <summary>
        /// Number of batches per epoch.
        /// </summary>
        public int NumBatches => NumSequences / _batchSize;

        /// <summary>
        /// Sequence length used for training.
        /// </summary>
        public int SeqLength => _seqLength;

        /// <summary>
        /// Batch size.
        /// </summary>
        public int BatchSize => _batchSize;

        /// <summary>
        /// Create a text dataset from pre-tokenized IDs.
        /// </summary>
        /// <param name="tokenIds">Array of token IDs.</param>
        /// <param name="seqLength">Sequence length for training.</param>
        /// <param name="batchSize">Batch size.</param>
        /// <param name="shuffle">Whether to shuffle sequences.</param>
        /// <param name="seed">Random seed for shuffling.</param>
        public TextDataset(int[] tokenIds, int seqLength, int batchSize, bool shuffle = true, int? seed = null)
        {
            if (tokenIds.Length < seqLength + 1)
                throw new ArgumentException($"Token array length {tokenIds.Length} is too short for sequence length {seqLength}");

            _tokenIds = tokenIds;
            _seqLength = seqLength;
            _batchSize = batchSize;
            _shuffle = shuffle;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();

            // Initialize sequence indices
            _indices = Enumerable.Range(0, NumSequences).ToArray();
            Reset();
        }

        /// <summary>
        /// Create a text dataset from raw text and a tokenizer.
        /// </summary>
        public static TextDataset FromText(string text, BPETokenizer tokenizer, int seqLength, int batchSize,
                                           bool shuffle = true, int? seed = null)
        {
            var tokenIds = tokenizer.Encode(text);
            return new TextDataset(tokenIds, seqLength, batchSize, shuffle, seed);
        }

        /// <summary>
        /// Create a text dataset from a text file.
        /// </summary>
        public static TextDataset FromFile(string filePath, BPETokenizer tokenizer, int seqLength, int batchSize,
                                           bool shuffle = true, int? seed = null)
        {
            string text = File.ReadAllText(filePath);
            return FromText(text, tokenizer, seqLength, batchSize, shuffle, seed);
        }

        /// <summary>
        /// Reset the dataset for a new epoch.
        /// </summary>
        public void Reset()
        {
            _currentPosition = 0;

            if (_shuffle)
            {
                // Fisher-Yates shuffle
                for (int i = _indices.Length - 1; i > 0; i--)
                {
                    int j = _random.Next(i + 1);
                    int temp = _indices[i];
                    _indices[i] = _indices[j];
                    _indices[j] = temp;
                }
            }
        }

        /// <summary>
        /// Check if there are more batches available.
        /// </summary>
        public bool HasNextBatch()
        {
            return _currentPosition + _batchSize <= NumSequences;
        }

        /// <summary>
        /// Get the next batch of input/target pairs.
        /// </summary>
        /// <returns>Tuple of (inputs, targets) where both are [batch, seqLen].</returns>
        public (int[,] inputs, int[,] targets) GetNextBatch()
        {
            if (!HasNextBatch())
                throw new InvalidOperationException("No more batches available. Call Reset() to start a new epoch.");

            var inputs = new int[_batchSize, _seqLength];
            var targets = new int[_batchSize, _seqLength];

            for (int b = 0; b < _batchSize; b++)
            {
                int startIdx = _indices[_currentPosition + b];

                for (int s = 0; s < _seqLength; s++)
                {
                    inputs[b, s] = _tokenIds[startIdx + s];
                    targets[b, s] = _tokenIds[startIdx + s + 1];  // Next token prediction
                }
            }

            _currentPosition += _batchSize;
            return (inputs, targets);
        }

        /// <summary>
        /// Get all batches for an epoch as an enumerable.
        /// </summary>
        public IEnumerable<(int[,] inputs, int[,] targets)> GetBatches()
        {
            Reset();
            while (HasNextBatch())
            {
                yield return GetNextBatch();
            }
        }

        /// <summary>
        /// Get a single random batch (useful for validation).
        /// </summary>
        public (int[,] inputs, int[,] targets) GetRandomBatch()
        {
            var inputs = new int[_batchSize, _seqLength];
            var targets = new int[_batchSize, _seqLength];

            for (int b = 0; b < _batchSize; b++)
            {
                int startIdx = _random.Next(NumSequences);

                for (int s = 0; s < _seqLength; s++)
                {
                    inputs[b, s] = _tokenIds[startIdx + s];
                    targets[b, s] = _tokenIds[startIdx + s + 1];
                }
            }

            return (inputs, targets);
        }

        /// <summary>
        /// Split the dataset into train and validation sets.
        /// </summary>
        /// <param name="validationRatio">Fraction of data to use for validation (0-1).</param>
        /// <returns>Tuple of (trainDataset, validationDataset).</returns>
        public (TextDataset train, TextDataset validation) Split(float validationRatio = 0.1f)
        {
            if (validationRatio <= 0 || validationRatio >= 1)
                throw new ArgumentException("Validation ratio must be between 0 and 1");

            int splitPoint = (int)(_tokenIds.Length * (1 - validationRatio));

            var trainTokens = _tokenIds.Take(splitPoint).ToArray();
            var valTokens = _tokenIds.Skip(splitPoint).ToArray();

            var trainDataset = new TextDataset(trainTokens, _seqLength, _batchSize, _shuffle);
            var valDataset = new TextDataset(valTokens, _seqLength, _batchSize, shuffle: false);

            return (trainDataset, valDataset);
        }

        /// <summary>
        /// Get dataset statistics.
        /// </summary>
        public string GetStats()
        {
            return $"TextDataset: {TotalTokens:N0} tokens, {NumSequences:N0} sequences, " +
                   $"{NumBatches:N0} batches (batch_size={_batchSize}, seq_len={_seqLength})";
        }
    }

    /// <summary>
    /// Utility class for creating training data from multiple text sources.
    /// </summary>
    public class TextDataLoader
    {
        private readonly BPETokenizer _tokenizer;
        private readonly List<int> _allTokens;

        public TextDataLoader(BPETokenizer tokenizer)
        {
            _tokenizer = tokenizer;
            _allTokens = new List<int>();
        }

        /// <summary>
        /// Add text to the dataset.
        /// </summary>
        public void AddText(string text, bool addBos = true, bool addEos = true)
        {
            var tokens = _tokenizer.Encode(text, addBos, addEos);
            _allTokens.AddRange(tokens);
        }

        /// <summary>
        /// Add text from a file.
        /// </summary>
        public void AddFile(string filePath, bool addBos = true, bool addEos = true)
        {
            string text = File.ReadAllText(filePath);
            AddText(text, addBos, addEos);
        }

        /// <summary>
        /// Add text from multiple files in a directory.
        /// </summary>
        public void AddDirectory(string directory, string pattern = "*.txt", bool addBos = true, bool addEos = true)
        {
            var files = Directory.GetFiles(directory, pattern);
            foreach (var file in files)
            {
                AddFile(file, addBos, addEos);
            }
            Console.WriteLine($"Added {files.Length} files from {directory}");
        }

        /// <summary>
        /// Create the dataset from all added texts.
        /// </summary>
        public TextDataset CreateDataset(int seqLength, int batchSize, bool shuffle = true, int? seed = null)
        {
            return new TextDataset(_allTokens.ToArray(), seqLength, batchSize, shuffle, seed);
        }

        /// <summary>
        /// Get total number of tokens loaded.
        /// </summary>
        public int TotalTokens => _allTokens.Count;
    }
}
