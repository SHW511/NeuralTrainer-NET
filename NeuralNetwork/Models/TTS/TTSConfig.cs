using System;
using System.IO;
using System.Text.Json;

namespace NeuralNetwork.Models.TTS
{
    /// <summary>
    /// Configuration for Text-to-Speech model.
    /// Tacotron-style encoder-attention-decoder architecture.
    /// </summary>
    public class TTSConfig
    {
        // ===== Text Encoder =====

        /// <summary>
        /// Size of character/phoneme vocabulary.
        /// </summary>
        public int VocabSize { get; set; } = 256;

        /// <summary>
        /// Embedding dimension for text characters.
        /// </summary>
        public int TextEmbeddingDim { get; set; } = 512;

        /// <summary>
        /// Number of encoder convolutional layers.
        /// </summary>
        public int EncoderConvLayers { get; set; } = 3;

        /// <summary>
        /// Kernel size for encoder convolutions.
        /// </summary>
        public int EncoderKernelSize { get; set; } = 5;

        /// <summary>
        /// Hidden dimension for encoder.
        /// </summary>
        public int EncoderDim { get; set; } = 512;

        // ===== Audio Output =====

        /// <summary>
        /// Number of mel spectrogram bins.
        /// </summary>
        public int MelBins { get; set; } = 80;

        /// <summary>
        /// Number of mel frames output per decoder step.
        /// Higher values speed up inference but may reduce quality.
        /// </summary>
        public int OutputsPerStep { get; set; } = 2;

        /// <summary>
        /// Maximum mel spectrogram length (frames).
        /// </summary>
        public int MaxMelLength { get; set; } = 1000;

        /// <summary>
        /// Maximum text length (characters).
        /// </summary>
        public int MaxTextLength { get; set; } = 200;

        // ===== Attention =====

        /// <summary>
        /// Attention dimension.
        /// </summary>
        public int AttentionDim { get; set; } = 128;

        /// <summary>
        /// Number of attention filters for location-sensitive attention.
        /// </summary>
        public int AttentionFilters { get; set; } = 32;

        /// <summary>
        /// Kernel size for attention location convolution.
        /// </summary>
        public int AttentionKernelSize { get; set; } = 31;

        // ===== Decoder =====

        /// <summary>
        /// Prenet dimensions (list of layer sizes).
        /// </summary>
        public int[] PrenetDims { get; set; } = { 256, 256 };

        /// <summary>
        /// Decoder LSTM dimension.
        /// </summary>
        public int DecoderDim { get; set; } = 1024;

        /// <summary>
        /// Number of decoder LSTM layers.
        /// </summary>
        public int DecoderLayers { get; set; } = 2;

        // ===== Postnet =====

        /// <summary>
        /// Number of postnet convolutional layers.
        /// </summary>
        public int PostnetLayers { get; set; } = 5;

        /// <summary>
        /// Postnet convolution kernel size.
        /// </summary>
        public int PostnetKernelSize { get; set; } = 5;

        /// <summary>
        /// Postnet convolution channels.
        /// </summary>
        public int PostnetChannels { get; set; } = 512;

        // ===== Speaker Embedding (for multi-speaker/voice cloning) =====

        /// <summary>
        /// Number of speakers (0 for single speaker).
        /// </summary>
        public int NumSpeakers { get; set; } = 0;

        /// <summary>
        /// Speaker embedding dimension.
        /// </summary>
        public int SpeakerEmbeddingDim { get; set; } = 256;

        // ===== Regularization =====

        /// <summary>
        /// Dropout rate.
        /// </summary>
        public float DropoutRate { get; set; } = 0.5f;

        /// <summary>
        /// Prenet dropout (typically higher than other dropout).
        /// </summary>
        public float PrenetDropout { get; set; } = 0.5f;

        // ===== Training =====

        /// <summary>
        /// Learning rate.
        /// </summary>
        public float LearningRate { get; set; } = 1e-3f;

        /// <summary>
        /// Weight decay.
        /// </summary>
        public float WeightDecay { get; set; } = 1e-6f;

        /// <summary>
        /// Gradient clipping value.
        /// </summary>
        public float GradientClip { get; set; } = 1.0f;

        // ===== Audio Processing =====

        /// <summary>
        /// Audio sample rate.
        /// </summary>
        public int SampleRate { get; set; } = 22050;

        /// <summary>
        /// FFT size.
        /// </summary>
        public int FFTSize { get; set; } = 1024;

        /// <summary>
        /// Hop length for spectrogram.
        /// </summary>
        public int HopLength { get; set; } = 256;

        // ===== Validation =====

        public void Validate()
        {
            if (VocabSize <= 0)
                throw new ArgumentException("VocabSize must be positive");

            if (TextEmbeddingDim <= 0)
                throw new ArgumentException("TextEmbeddingDim must be positive");

            if (MelBins <= 0)
                throw new ArgumentException("MelBins must be positive");

            if (OutputsPerStep <= 0)
                throw new ArgumentException("OutputsPerStep must be positive");

            if (AttentionDim <= 0)
                throw new ArgumentException("AttentionDim must be positive");

            if (DecoderDim <= 0)
                throw new ArgumentException("DecoderDim must be positive");

            if (DropoutRate < 0 || DropoutRate >= 1)
                throw new ArgumentException("DropoutRate must be in [0, 1)");

            if (LearningRate <= 0)
                throw new ArgumentException("LearningRate must be positive");
        }

        // ===== Factory Methods =====

        /// <summary>
        /// Small model for testing.
        /// </summary>
        public static TTSConfig Small()
        {
            return new TTSConfig
            {
                TextEmbeddingDim = 128,
                EncoderConvLayers = 2,
                EncoderDim = 128,
                AttentionDim = 64,
                AttentionFilters = 16,
                PrenetDims = new[] { 64, 64 },
                DecoderDim = 256,
                DecoderLayers = 1,
                PostnetLayers = 3,
                PostnetChannels = 128,
                DropoutRate = 0.1f
            };
        }

        /// <summary>
        /// Standard model for production use.
        /// </summary>
        public static TTSConfig Standard()
        {
            return new TTSConfig(); // Default values
        }

        /// <summary>
        /// Large model for high quality synthesis.
        /// </summary>
        public static TTSConfig Large()
        {
            return new TTSConfig
            {
                TextEmbeddingDim = 512,
                EncoderConvLayers = 5,
                EncoderDim = 512,
                AttentionDim = 256,
                AttentionFilters = 64,
                PrenetDims = new[] { 256, 256 },
                DecoderDim = 1024,
                DecoderLayers = 2,
                PostnetLayers = 5,
                PostnetChannels = 512,
                SpeakerEmbeddingDim = 512
            };
        }

        // ===== Serialization =====

        public void Save(string path)
        {
            var options = new JsonSerializerOptions { WriteIndented = true };
            string json = JsonSerializer.Serialize(this, options);
            File.WriteAllText(path, json);
        }

        public static TTSConfig Load(string path)
        {
            string json = File.ReadAllText(path);
            var config = JsonSerializer.Deserialize<TTSConfig>(json);
            config.Validate();
            return config;
        }

        public TTSConfig Clone()
        {
            var json = JsonSerializer.Serialize(this);
            return JsonSerializer.Deserialize<TTSConfig>(json);
        }

        public override string ToString()
        {
            return $"TTSConfig(vocab={VocabSize}, mel={MelBins}, enc={EncoderDim}, " +
                   $"dec={DecoderDim}x{DecoderLayers}, speakers={NumSpeakers})";
        }
    }
}
