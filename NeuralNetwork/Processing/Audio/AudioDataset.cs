using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork.Processing.Audio
{
    /// <summary>
    /// Dataset for TTS training containing text-audio pairs.
    /// Supports loading from directories with audio files and transcription files.
    /// </summary>
    public class AudioDataset
    {
        public List<AudioTextPair> Samples { get; private set; } = new();
        public int NumSamples => Samples.Count;
        public int BatchSize { get; set; } = 16;
        public int NumBatches => (NumSamples + BatchSize - 1) / BatchSize;

        private AudioPreProcessing _audioProcessor;
        private MelSpectrogram _melProcessor;
        private Random _random;
        private int[] _shuffledIndices;
        private int _currentBatch;

        public AudioDataset(int batchSize = 16, int seed = 42)
        {
            BatchSize = batchSize;
            _random = new Random(seed);
            _audioProcessor = new AudioPreProcessing();
            _melProcessor = new MelSpectrogram();
        }

        /// <summary>
        /// Load dataset from a directory containing WAV files and a metadata CSV.
        /// Expected format: metadata.csv with lines "filename|transcription|speaker_id"
        /// </summary>
        public static AudioDataset FromDirectory(string path, int batchSize = 16, int seed = 42)
        {
            var dataset = new AudioDataset(batchSize, seed);

            string metadataPath = Path.Combine(path, "metadata.csv");
            if (!File.Exists(metadataPath))
            {
                // Try to find WAV files and use filenames as transcriptions
                var wavFiles = Directory.GetFiles(path, "*.wav", SearchOption.AllDirectories);
                foreach (var wavFile in wavFiles)
                {
                    dataset.Samples.Add(new AudioTextPair
                    {
                        AudioPath = wavFile,
                        Text = Path.GetFileNameWithoutExtension(wavFile),
                        SpeakerId = 0
                    });
                }
            }
            else
            {
                // Load from metadata file
                var lines = File.ReadAllLines(metadataPath);
                foreach (var line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;

                    var parts = line.Split('|');
                    if (parts.Length >= 2)
                    {
                        string audioPath = Path.Combine(path, parts[0].Trim());
                        if (!audioPath.EndsWith(".wav")) audioPath += ".wav";

                        dataset.Samples.Add(new AudioTextPair
                        {
                            AudioPath = audioPath,
                            Text = parts[1].Trim(),
                            SpeakerId = parts.Length > 2 ? int.Parse(parts[2].Trim()) : 0
                        });
                    }
                }
            }

            dataset.Shuffle();
            return dataset;
        }

        /// <summary>
        /// Create dataset from explicit list of audio-text pairs.
        /// </summary>
        public static AudioDataset FromPairs(IEnumerable<(string audioPath, string text)> pairs,
            int batchSize = 16, int seed = 42)
        {
            var dataset = new AudioDataset(batchSize, seed);

            foreach (var (audioPath, text) in pairs)
            {
                dataset.Samples.Add(new AudioTextPair
                {
                    AudioPath = audioPath,
                    Text = text,
                    SpeakerId = 0
                });
            }

            dataset.Shuffle();
            return dataset;
        }

        /// <summary>
        /// Shuffle the dataset.
        /// </summary>
        public void Shuffle()
        {
            _shuffledIndices = Enumerable.Range(0, NumSamples).ToArray();
            for (int i = _shuffledIndices.Length - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (_shuffledIndices[i], _shuffledIndices[j]) = (_shuffledIndices[j], _shuffledIndices[i]);
            }
            _currentBatch = 0;
        }

        /// <summary>
        /// Get the next batch of training data.
        /// Returns (texts, mel_spectrograms, speaker_ids).
        /// </summary>
        public AudioBatch GetNextBatch()
        {
            if (_shuffledIndices == null) Shuffle();

            int start = _currentBatch * BatchSize;
            int end = Math.Min(start + BatchSize, NumSamples);
            int actualBatchSize = end - start;

            var batch = new AudioBatch
            {
                Texts = new string[actualBatchSize],
                SpeakerIds = new int[actualBatchSize]
            };

            // Load audio and compute mel spectrograms
            var melSpecs = new List<float[,]>();
            int maxMelLength = 0;

            for (int i = 0; i < actualBatchSize; i++)
            {
                int idx = _shuffledIndices[start + i];
                var sample = Samples[idx];

                batch.Texts[i] = sample.Text;
                batch.SpeakerIds[i] = sample.SpeakerId;

                // Load and process audio
                if (File.Exists(sample.AudioPath))
                {
                    float[] audio = _audioProcessor.LoadWav(sample.AudioPath);
                    audio = _audioProcessor.ApplyPreEmphasis(audio);
                    float[,] mel = _melProcessor.WaveformToMelSpectrogram(audio);
                    melSpecs.Add(mel);

                    if (mel.GetLength(0) > maxMelLength)
                        maxMelLength = mel.GetLength(0);
                }
                else
                {
                    // Create empty placeholder
                    melSpecs.Add(new float[1, _melProcessor.MelBins]);
                }
            }

            // Pad mel spectrograms to same length
            int melBins = _melProcessor.MelBins;
            batch.MelSpectrograms = new float[actualBatchSize, maxMelLength, melBins];
            batch.MelLengths = new int[actualBatchSize];

            for (int i = 0; i < actualBatchSize; i++)
            {
                var mel = melSpecs[i];
                int len = mel.GetLength(0);
                batch.MelLengths[i] = len;

                for (int t = 0; t < len; t++)
                {
                    for (int m = 0; m < melBins; m++)
                    {
                        batch.MelSpectrograms[i, t, m] = mel[t, m];
                    }
                }
            }

            _currentBatch++;
            if (_currentBatch >= NumBatches)
            {
                _currentBatch = 0;
            }

            return batch;
        }

        /// <summary>
        /// Reset to beginning of dataset.
        /// </summary>
        public void Reset()
        {
            _currentBatch = 0;
        }

        /// <summary>
        /// Get iterator over all batches.
        /// </summary>
        public IEnumerable<AudioBatch> GetBatches()
        {
            Reset();
            for (int i = 0; i < NumBatches; i++)
            {
                yield return GetNextBatch();
            }
        }

        /// <summary>
        /// Split dataset into training and validation sets.
        /// </summary>
        public (AudioDataset train, AudioDataset val) Split(float valRatio = 0.1f)
        {
            int valCount = (int)(NumSamples * valRatio);
            int trainCount = NumSamples - valCount;

            var trainDataset = new AudioDataset(BatchSize, _random.Next());
            var valDataset = new AudioDataset(BatchSize, _random.Next());

            Shuffle();

            for (int i = 0; i < trainCount; i++)
            {
                trainDataset.Samples.Add(Samples[_shuffledIndices[i]]);
            }
            for (int i = trainCount; i < NumSamples; i++)
            {
                valDataset.Samples.Add(Samples[_shuffledIndices[i]]);
            }

            trainDataset.Shuffle();
            valDataset.Shuffle();

            return (trainDataset, valDataset);
        }
    }

    /// <summary>
    /// Single audio-text pair for TTS training.
    /// </summary>
    public class AudioTextPair
    {
        public string AudioPath { get; set; }
        public string Text { get; set; }
        public int SpeakerId { get; set; }

        // Cached processed data
        public float[] AudioSamples { get; set; }
        public float[,] MelSpectrogram { get; set; }
        public int[] TextTokens { get; set; }
    }

    /// <summary>
    /// Batch of audio-text data for training.
    /// </summary>
    public class AudioBatch
    {
        public string[] Texts { get; set; }
        public int[] SpeakerIds { get; set; }
        public float[,,] MelSpectrograms { get; set; }  // [batch, time, mel_bins]
        public int[] MelLengths { get; set; }            // Actual length of each mel spectrogram
        public int[][] TextTokens { get; set; }          // Tokenized text (set by model)
        public int[] TextLengths { get; set; }           // Actual length of each text
    }
}
