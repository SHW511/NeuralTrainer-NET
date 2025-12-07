using System;
using System.IO;

namespace NeuralNetwork.Processing.Audio
{
    /// <summary>
    /// Audio preprocessing utilities for loading, resampling, and normalizing audio data.
    /// </summary>
    public class AudioPreProcessing
    {
        public int SampleRate { get; set; } = 22050;
        public int TargetSampleRate { get; set; } = 22050;
        public bool Normalize { get; set; } = true;
        public float PreEmphasis { get; set; } = 0.97f;

        /// <summary>
        /// Load a WAV file and return the audio samples as float array.
        /// </summary>
        public float[] LoadWav(string path)
        {
            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);

            // Read RIFF header
            string riff = new string(reader.ReadChars(4));
            if (riff != "RIFF")
                throw new InvalidDataException($"Not a valid WAV file: {path}");

            reader.ReadInt32(); // File size
            string wave = new string(reader.ReadChars(4));
            if (wave != "WAVE")
                throw new InvalidDataException($"Not a valid WAV file: {path}");

            // Find fmt chunk
            int channels = 0;
            int sampleRate = 0;
            int bitsPerSample = 0;

            while (stream.Position < stream.Length)
            {
                string chunkId = new string(reader.ReadChars(4));
                int chunkSize = reader.ReadInt32();

                if (chunkId == "fmt ")
                {
                    short audioFormat = reader.ReadInt16();
                    channels = reader.ReadInt16();
                    sampleRate = reader.ReadInt32();
                    reader.ReadInt32(); // Byte rate
                    reader.ReadInt16(); // Block align
                    bitsPerSample = reader.ReadInt16();

                    // Skip any extra format bytes
                    if (chunkSize > 16)
                        reader.ReadBytes(chunkSize - 16);
                }
                else if (chunkId == "data")
                {
                    // Read audio data
                    int numSamples = chunkSize / (bitsPerSample / 8) / channels;
                    float[] samples = new float[numSamples];

                    for (int i = 0; i < numSamples; i++)
                    {
                        float sample = 0;

                        // Read all channels and average (convert to mono)
                        for (int c = 0; c < channels; c++)
                        {
                            if (bitsPerSample == 16)
                            {
                                short s = reader.ReadInt16();
                                sample += s / 32768f;
                            }
                            else if (bitsPerSample == 8)
                            {
                                byte b = reader.ReadByte();
                                sample += (b - 128) / 128f;
                            }
                            else if (bitsPerSample == 24)
                            {
                                byte b1 = reader.ReadByte();
                                byte b2 = reader.ReadByte();
                                byte b3 = reader.ReadByte();
                                int s = (b3 << 16) | (b2 << 8) | b1;
                                if ((s & 0x800000) != 0) s |= unchecked((int)0xFF000000);
                                sample += s / 8388608f;
                            }
                            else if (bitsPerSample == 32)
                            {
                                int s = reader.ReadInt32();
                                sample += s / 2147483648f;
                            }
                        }

                        samples[i] = sample / channels;
                    }

                    SampleRate = sampleRate;

                    // Resample if needed
                    if (sampleRate != TargetSampleRate)
                        samples = Resample(samples, sampleRate, TargetSampleRate);

                    // Normalize if enabled
                    if (Normalize)
                        samples = NormalizeAudio(samples);

                    return samples;
                }
                else
                {
                    // Skip unknown chunk
                    reader.ReadBytes(chunkSize);
                }
            }

            throw new InvalidDataException($"No data chunk found in WAV file: {path}");
        }

        /// <summary>
        /// Save audio samples to a WAV file.
        /// </summary>
        public void SaveWav(string path, float[] samples, int sampleRate = 0)
        {
            if (sampleRate == 0) sampleRate = TargetSampleRate;

            using var stream = File.Create(path);
            using var writer = new BinaryWriter(stream);

            int bitsPerSample = 16;
            int channels = 1;
            int dataSize = samples.Length * (bitsPerSample / 8);

            // RIFF header
            writer.Write("RIFF".ToCharArray());
            writer.Write(36 + dataSize);
            writer.Write("WAVE".ToCharArray());

            // fmt chunk
            writer.Write("fmt ".ToCharArray());
            writer.Write(16); // Chunk size
            writer.Write((short)1); // Audio format (PCM)
            writer.Write((short)channels);
            writer.Write(sampleRate);
            writer.Write(sampleRate * channels * (bitsPerSample / 8)); // Byte rate
            writer.Write((short)(channels * (bitsPerSample / 8))); // Block align
            writer.Write((short)bitsPerSample);

            // data chunk
            writer.Write("data".ToCharArray());
            writer.Write(dataSize);

            // Write samples
            foreach (float sample in samples)
            {
                float clamped = Math.Clamp(sample, -1f, 1f);
                short s = (short)(clamped * 32767);
                writer.Write(s);
            }
        }

        /// <summary>
        /// Resample audio from one sample rate to another using linear interpolation.
        /// </summary>
        public float[] Resample(float[] samples, int fromRate, int toRate)
        {
            if (fromRate == toRate) return samples;

            double ratio = (double)toRate / fromRate;
            int newLength = (int)(samples.Length * ratio);
            float[] resampled = new float[newLength];

            for (int i = 0; i < newLength; i++)
            {
                double srcIndex = i / ratio;
                int srcIndexInt = (int)srcIndex;
                double frac = srcIndex - srcIndexInt;

                if (srcIndexInt + 1 < samples.Length)
                {
                    resampled[i] = (float)(samples[srcIndexInt] * (1 - frac) +
                                          samples[srcIndexInt + 1] * frac);
                }
                else
                {
                    resampled[i] = samples[srcIndexInt];
                }
            }

            return resampled;
        }

        /// <summary>
        /// Normalize audio to have maximum absolute value of 1.
        /// </summary>
        public float[] NormalizeAudio(float[] samples)
        {
            float maxAbs = 0;
            for (int i = 0; i < samples.Length; i++)
            {
                float abs = Math.Abs(samples[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            if (maxAbs < 1e-6f) return samples;

            float[] normalized = new float[samples.Length];
            float scale = 1f / maxAbs;
            for (int i = 0; i < samples.Length; i++)
            {
                normalized[i] = samples[i] * scale;
            }

            return normalized;
        }

        /// <summary>
        /// Apply pre-emphasis filter to enhance high frequencies.
        /// </summary>
        public float[] ApplyPreEmphasis(float[] samples)
        {
            float[] result = new float[samples.Length];
            result[0] = samples[0];

            for (int i = 1; i < samples.Length; i++)
            {
                result[i] = samples[i] - PreEmphasis * samples[i - 1];
            }

            return result;
        }

        /// <summary>
        /// Remove pre-emphasis (inverse filter).
        /// </summary>
        public float[] RemovePreEmphasis(float[] samples)
        {
            float[] result = new float[samples.Length];
            result[0] = samples[0];

            for (int i = 1; i < samples.Length; i++)
            {
                result[i] = samples[i] + PreEmphasis * result[i - 1];
            }

            return result;
        }

        /// <summary>
        /// Trim silence from beginning and end of audio.
        /// </summary>
        public float[] TrimSilence(float[] samples, float threshold = 0.01f, int minSilenceSamples = 1000)
        {
            int start = 0;
            int end = samples.Length - 1;

            // Find start (first sample above threshold)
            int silenceCount = 0;
            for (int i = 0; i < samples.Length; i++)
            {
                if (Math.Abs(samples[i]) > threshold)
                {
                    silenceCount = 0;
                    start = Math.Max(0, i - minSilenceSamples / 2);
                    break;
                }
                silenceCount++;
            }

            // Find end (last sample above threshold)
            silenceCount = 0;
            for (int i = samples.Length - 1; i >= 0; i--)
            {
                if (Math.Abs(samples[i]) > threshold)
                {
                    end = Math.Min(samples.Length - 1, i + minSilenceSamples / 2);
                    break;
                }
                silenceCount++;
            }

            if (start >= end) return samples;

            float[] trimmed = new float[end - start + 1];
            Array.Copy(samples, start, trimmed, 0, trimmed.Length);
            return trimmed;
        }

        /// <summary>
        /// Split audio into fixed-length chunks with optional overlap.
        /// </summary>
        public float[][] SplitIntoChunks(float[] samples, int chunkSize, int hopSize = 0)
        {
            if (hopSize == 0) hopSize = chunkSize;

            int numChunks = (samples.Length - chunkSize) / hopSize + 1;
            if (numChunks < 1) numChunks = 1;

            float[][] chunks = new float[numChunks][];

            for (int i = 0; i < numChunks; i++)
            {
                int start = i * hopSize;
                int size = Math.Min(chunkSize, samples.Length - start);
                chunks[i] = new float[chunkSize];

                Array.Copy(samples, start, chunks[i], 0, size);
                // Zero-pad if needed
            }

            return chunks;
        }

        /// <summary>
        /// Concatenate audio chunks back together with overlap-add.
        /// </summary>
        public float[] ConcatenateChunks(float[][] chunks, int hopSize)
        {
            if (chunks.Length == 0) return Array.Empty<float>();

            int chunkSize = chunks[0].Length;
            int totalLength = (chunks.Length - 1) * hopSize + chunkSize;
            float[] result = new float[totalLength];
            float[] weights = new float[totalLength];

            for (int i = 0; i < chunks.Length; i++)
            {
                int start = i * hopSize;
                for (int j = 0; j < chunks[i].Length && start + j < totalLength; j++)
                {
                    result[start + j] += chunks[i][j];
                    weights[start + j] += 1;
                }
            }

            // Normalize by overlap count
            for (int i = 0; i < totalLength; i++)
            {
                if (weights[i] > 0)
                    result[i] /= weights[i];
            }

            return result;
        }
    }
}
