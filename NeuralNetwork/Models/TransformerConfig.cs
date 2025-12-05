using System;
using System.IO;
using System.Text.Json;

namespace NeuralNetwork.Models
{
    /// <summary>
    /// Configuration class for Transformer Language Model.
    /// Contains all hyperparameters for model architecture and training.
    /// </summary>
    public class TransformerConfig
    {
        // ===== Model Architecture =====

        /// <summary>
        /// Vocabulary size (number of unique tokens).
        /// </summary>
        public int VocabSize { get; set; } = 32000;

        /// <summary>
        /// Maximum sequence length the model can handle.
        /// </summary>
        public int MaxSeqLen { get; set; } = 512;

        /// <summary>
        /// Number of Transformer blocks (layers).
        /// </summary>
        public int NumLayers { get; set; } = 6;

        /// <summary>
        /// Number of attention heads per layer.
        /// </summary>
        public int NumHeads { get; set; } = 8;

        /// <summary>
        /// Model dimension (d_model). Must be divisible by NumHeads.
        /// </summary>
        public int EmbeddingDim { get; set; } = 512;

        /// <summary>
        /// Feed-forward network hidden dimension. Typically 4 * EmbeddingDim.
        /// </summary>
        public int FFNDim { get; set; } = 2048;

        // ===== Regularization =====

        /// <summary>
        /// Dropout rate for residual connections.
        /// </summary>
        public float DropoutRate { get; set; } = 0.1f;

        /// <summary>
        /// Dropout rate for attention weights.
        /// </summary>
        public float AttentionDropout { get; set; } = 0.1f;

        /// <summary>
        /// Dropout rate for embeddings.
        /// </summary>
        public float EmbeddingDropout { get; set; } = 0.1f;

        // ===== Training =====

        /// <summary>
        /// Base learning rate.
        /// </summary>
        public float LearningRate { get; set; } = 1e-4f;

        /// <summary>
        /// Number of warmup steps for learning rate schedule.
        /// </summary>
        public int WarmupSteps { get; set; } = 4000;

        /// <summary>
        /// Weight decay coefficient (L2 regularization).
        /// </summary>
        public float WeightDecay { get; set; } = 0.01f;

        /// <summary>
        /// Gradient clipping value. Set to 0 to disable.
        /// </summary>
        public float GradientClip { get; set; } = 1.0f;

        /// <summary>
        /// Adam optimizer beta1.
        /// </summary>
        public float Beta1 { get; set; } = 0.9f;

        /// <summary>
        /// Adam optimizer beta2.
        /// </summary>
        public float Beta2 { get; set; } = 0.999f;

        /// <summary>
        /// Adam optimizer epsilon.
        /// </summary>
        public float Epsilon { get; set; } = 1e-8f;

        // ===== Architecture Options =====

        /// <summary>
        /// Whether to tie embedding and output projection weights.
        /// Reduces parameters and often improves performance.
        /// </summary>
        public bool TieEmbeddings { get; set; } = true;

        /// <summary>
        /// Whether to use causal (autoregressive) masking.
        /// </summary>
        public bool UseCausalMask { get; set; } = true;

        /// <summary>
        /// Layer normalization type: "pre" (before sublayer) or "post" (after sublayer).
        /// Pre-LN is more stable for training.
        /// </summary>
        public string NormType { get; set; } = "pre";

        /// <summary>
        /// Whether to use learnable positional embeddings instead of sinusoidal.
        /// </summary>
        public bool LearnablePositionalEncoding { get; set; } = false;

        // ===== Computed Properties =====

        /// <summary>
        /// Dimension of each attention head.
        /// </summary>
        public int HeadDim => EmbeddingDim / NumHeads;

        /// <summary>
        /// Validate the configuration.
        /// </summary>
        public void Validate()
        {
            if (VocabSize <= 0)
                throw new ArgumentException("VocabSize must be positive");

            if (MaxSeqLen <= 0)
                throw new ArgumentException("MaxSeqLen must be positive");

            if (NumLayers <= 0)
                throw new ArgumentException("NumLayers must be positive");

            if (NumHeads <= 0)
                throw new ArgumentException("NumHeads must be positive");

            if (EmbeddingDim <= 0)
                throw new ArgumentException("EmbeddingDim must be positive");

            if (EmbeddingDim % NumHeads != 0)
                throw new ArgumentException($"EmbeddingDim ({EmbeddingDim}) must be divisible by NumHeads ({NumHeads})");

            if (FFNDim <= 0)
                throw new ArgumentException("FFNDim must be positive");

            if (DropoutRate < 0 || DropoutRate >= 1)
                throw new ArgumentException("DropoutRate must be in [0, 1)");

            if (LearningRate <= 0)
                throw new ArgumentException("LearningRate must be positive");

            if (NormType != "pre" && NormType != "post")
                throw new ArgumentException("NormType must be 'pre' or 'post'");
        }

        /// <summary>
        /// Create a small configuration for testing/debugging.
        /// </summary>
        public static TransformerConfig Small()
        {
            return new TransformerConfig
            {
                VocabSize = 1000,
                MaxSeqLen = 128,
                NumLayers = 2,
                NumHeads = 4,
                EmbeddingDim = 128,
                FFNDim = 512,
                DropoutRate = 0.1f
            };
        }

        /// <summary>
        /// Create a medium configuration similar to GPT-2 Small.
        /// </summary>
        public static TransformerConfig Medium()
        {
            return new TransformerConfig
            {
                VocabSize = 50257,
                MaxSeqLen = 1024,
                NumLayers = 12,
                NumHeads = 12,
                EmbeddingDim = 768,
                FFNDim = 3072,
                DropoutRate = 0.1f
            };
        }

        /// <summary>
        /// Create a large configuration similar to GPT-2 Medium.
        /// </summary>
        public static TransformerConfig Large()
        {
            return new TransformerConfig
            {
                VocabSize = 50257,
                MaxSeqLen = 1024,
                NumLayers = 24,
                NumHeads = 16,
                EmbeddingDim = 1024,
                FFNDim = 4096,
                DropoutRate = 0.1f
            };
        }

        /// <summary>
        /// Save configuration to JSON file.
        /// </summary>
        public void Save(string path)
        {
            var options = new JsonSerializerOptions { WriteIndented = true };
            string json = JsonSerializer.Serialize(this, options);
            File.WriteAllText(path, json);
        }

        /// <summary>
        /// Load configuration from JSON file.
        /// </summary>
        public static TransformerConfig Load(string path)
        {
            string json = File.ReadAllText(path);
            var config = JsonSerializer.Deserialize<TransformerConfig>(json);
            config.Validate();
            return config;
        }

        /// <summary>
        /// Create a copy of this configuration.
        /// </summary>
        public TransformerConfig Clone()
        {
            return new TransformerConfig
            {
                VocabSize = VocabSize,
                MaxSeqLen = MaxSeqLen,
                NumLayers = NumLayers,
                NumHeads = NumHeads,
                EmbeddingDim = EmbeddingDim,
                FFNDim = FFNDim,
                DropoutRate = DropoutRate,
                AttentionDropout = AttentionDropout,
                EmbeddingDropout = EmbeddingDropout,
                LearningRate = LearningRate,
                WarmupSteps = WarmupSteps,
                WeightDecay = WeightDecay,
                GradientClip = GradientClip,
                Beta1 = Beta1,
                Beta2 = Beta2,
                Epsilon = Epsilon,
                TieEmbeddings = TieEmbeddings,
                UseCausalMask = UseCausalMask,
                NormType = NormType,
                LearnablePositionalEncoding = LearnablePositionalEncoding
            };
        }

        public override string ToString()
        {
            return $"TransformerConfig(vocab={VocabSize}, layers={NumLayers}, heads={NumHeads}, " +
                   $"dim={EmbeddingDim}, ffn={FFNDim}, maxSeq={MaxSeqLen})";
        }
    }
}
