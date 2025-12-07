using System;
using System.Numerics;

namespace NeuralNetwork.Processing.Audio
{
    /// <summary>
    /// Mel spectrogram computation using STFT and Mel filterbanks.
    /// Used as input representation for TTS and voice models.
    /// </summary>
    public class MelSpectrogram
    {
        public int SampleRate { get; set; } = 22050;
        public int FFTSize { get; set; } = 1024;
        public int HopLength { get; set; } = 256;
        public int WinLength { get; set; } = 1024;
        public int MelBins { get; set; } = 80;
        public float FMin { get; set; } = 0f;
        public float FMax { get; set; } = 8000f;
        public float RefLevel { get; set; } = 20f;
        public float MinLevel { get; set; } = -100f;

        private float[] _window;
        private float[,] _melFilterbank;

        public MelSpectrogram()
        {
            Initialize();
        }

        public MelSpectrogram(int sampleRate = 22050, int fftSize = 1024, int hopLength = 256, int melBins = 80)
        {
            SampleRate = sampleRate;
            FFTSize = fftSize;
            HopLength = hopLength;
            WinLength = fftSize;
            MelBins = melBins;
            FMax = sampleRate / 2f;
            Initialize();
        }

        private void Initialize()
        {
            _window = CreateHannWindow(WinLength);
            _melFilterbank = CreateMelFilterbank();
        }

        /// <summary>
        /// Convert audio waveform to mel spectrogram.
        /// Returns [time, mel_bins] array.
        /// </summary>
        public float[,] WaveformToMelSpectrogram(float[] waveform)
        {
            // Compute STFT magnitude
            var stft = ComputeSTFT(waveform);
            int numFrames = stft.GetLength(0);
            int numBins = stft.GetLength(1);

            // Compute magnitude spectrum
            float[,] magnitude = new float[numFrames, numBins];
            for (int t = 0; t < numFrames; t++)
            {
                for (int f = 0; f < numBins; f++)
                {
                    magnitude[t, f] = (float)stft[t, f].Magnitude;
                }
            }

            // Apply mel filterbank
            float[,] melSpec = ApplyMelFilterbank(magnitude);

            // Convert to log scale (dB)
            for (int t = 0; t < melSpec.GetLength(0); t++)
            {
                for (int m = 0; m < melSpec.GetLength(1); m++)
                {
                    float val = melSpec[t, m];
                    val = Math.Max(val, 1e-10f);
                    val = RefLevel * (float)Math.Log10(val);
                    val = Math.Max(val, MinLevel);
                    melSpec[t, m] = val;
                }
            }

            return melSpec;
        }

        /// <summary>
        /// Convert mel spectrogram back to waveform using Griffin-Lim algorithm.
        /// </summary>
        public float[] MelSpectrogramToWaveform(float[,] melSpec, int iterations = 60)
        {
            int numFrames = melSpec.GetLength(0);
            int numMels = melSpec.GetLength(1);
            int numBins = FFTSize / 2 + 1;

            // Convert from dB to linear
            float[,] melLinear = new float[numFrames, numMels];
            for (int t = 0; t < numFrames; t++)
            {
                for (int m = 0; m < numMels; m++)
                {
                    float db = melSpec[t, m];
                    melLinear[t, m] = (float)Math.Pow(10, db / RefLevel);
                }
            }

            // Invert mel filterbank (pseudo-inverse)
            float[,] magnitude = InvertMelFilterbank(melLinear);

            // Griffin-Lim algorithm to recover phase
            return GriffinLim(magnitude, iterations);
        }

        /// <summary>
        /// Compute Short-Time Fourier Transform.
        /// </summary>
        public Complex[,] ComputeSTFT(float[] waveform)
        {
            int padLength = FFTSize / 2;
            float[] padded = new float[waveform.Length + 2 * padLength];
            Array.Copy(waveform, 0, padded, padLength, waveform.Length);

            int numFrames = (padded.Length - FFTSize) / HopLength + 1;
            int numBins = FFTSize / 2 + 1;

            Complex[,] stft = new Complex[numFrames, numBins];
            Complex[] frame = new Complex[FFTSize];

            for (int t = 0; t < numFrames; t++)
            {
                int start = t * HopLength;

                // Apply window
                for (int i = 0; i < FFTSize; i++)
                {
                    if (i < WinLength && start + i < padded.Length)
                        frame[i] = new Complex(padded[start + i] * _window[i], 0);
                    else
                        frame[i] = Complex.Zero;
                }

                // Compute FFT
                FFT(frame);

                // Store positive frequencies
                for (int f = 0; f < numBins; f++)
                {
                    stft[t, f] = frame[f];
                }
            }

            return stft;
        }

        /// <summary>
        /// Compute inverse STFT to reconstruct waveform.
        /// </summary>
        public float[] ComputeISTFT(Complex[,] stft)
        {
            int numFrames = stft.GetLength(0);
            int numBins = stft.GetLength(1);

            int outputLength = (numFrames - 1) * HopLength + FFTSize;
            float[] output = new float[outputLength];
            float[] windowSum = new float[outputLength];

            Complex[] frame = new Complex[FFTSize];

            for (int t = 0; t < numFrames; t++)
            {
                // Build full spectrum (with conjugate symmetry)
                for (int f = 0; f < numBins; f++)
                {
                    frame[f] = stft[t, f];
                }
                for (int f = numBins; f < FFTSize; f++)
                {
                    frame[f] = Complex.Conjugate(stft[t, FFTSize - f]);
                }

                // Compute IFFT
                IFFT(frame);

                // Overlap-add with window
                int start = t * HopLength;
                for (int i = 0; i < FFTSize && start + i < outputLength; i++)
                {
                    float win = i < WinLength ? _window[i] : 0;
                    output[start + i] += (float)frame[i].Real * win;
                    windowSum[start + i] += win * win;
                }
            }

            // Normalize by window sum
            for (int i = 0; i < outputLength; i++)
            {
                if (windowSum[i] > 1e-8f)
                    output[i] /= windowSum[i];
            }

            // Remove padding
            int padLength = FFTSize / 2;
            int trimmedLength = Math.Max(0, outputLength - 2 * padLength);
            float[] trimmed = new float[trimmedLength];
            Array.Copy(output, padLength, trimmed, 0, trimmedLength);

            return trimmed;
        }

        /// <summary>
        /// Griffin-Lim algorithm for phase reconstruction.
        /// </summary>
        private float[] GriffinLim(float[,] magnitude, int iterations)
        {
            int numFrames = magnitude.GetLength(0);
            int numBins = magnitude.GetLength(1);

            // Initialize with random phase
            Random rand = new Random(42);
            Complex[,] stft = new Complex[numFrames, numBins];

            for (int t = 0; t < numFrames; t++)
            {
                for (int f = 0; f < numBins; f++)
                {
                    double phase = rand.NextDouble() * 2 * Math.PI;
                    stft[t, f] = Complex.FromPolarCoordinates(magnitude[t, f], phase);
                }
            }

            // Iterate
            for (int iter = 0; iter < iterations; iter++)
            {
                // ISTFT to get waveform
                float[] waveform = ComputeISTFT(stft);

                // STFT to get new phase
                Complex[,] newStft = ComputeSTFT(waveform);

                // Keep original magnitude, use new phase
                for (int t = 0; t < numFrames; t++)
                {
                    for (int f = 0; f < numBins; f++)
                    {
                        double phase = newStft[t, f].Phase;
                        stft[t, f] = Complex.FromPolarCoordinates(magnitude[t, f], phase);
                    }
                }
            }

            return ComputeISTFT(stft);
        }

        /// <summary>
        /// Apply mel filterbank to magnitude spectrum.
        /// </summary>
        private float[,] ApplyMelFilterbank(float[,] magnitude)
        {
            int numFrames = magnitude.GetLength(0);
            int numBins = magnitude.GetLength(1);

            float[,] melSpec = new float[numFrames, MelBins];

            for (int t = 0; t < numFrames; t++)
            {
                for (int m = 0; m < MelBins; m++)
                {
                    float sum = 0;
                    for (int f = 0; f < numBins; f++)
                    {
                        sum += magnitude[t, f] * _melFilterbank[m, f];
                    }
                    melSpec[t, m] = sum;
                }
            }

            return melSpec;
        }

        /// <summary>
        /// Invert mel filterbank (approximate).
        /// </summary>
        private float[,] InvertMelFilterbank(float[,] melSpec)
        {
            int numFrames = melSpec.GetLength(0);
            int numBins = FFTSize / 2 + 1;

            float[,] magnitude = new float[numFrames, numBins];

            // Compute pseudo-inverse by transposing and normalizing
            float[] filterSums = new float[numBins];
            for (int f = 0; f < numBins; f++)
            {
                for (int m = 0; m < MelBins; m++)
                {
                    filterSums[f] += _melFilterbank[m, f];
                }
                filterSums[f] = Math.Max(filterSums[f], 1e-10f);
            }

            for (int t = 0; t < numFrames; t++)
            {
                for (int f = 0; f < numBins; f++)
                {
                    float sum = 0;
                    for (int m = 0; m < MelBins; m++)
                    {
                        sum += melSpec[t, m] * _melFilterbank[m, f];
                    }
                    magnitude[t, f] = sum / filterSums[f];
                }
            }

            return magnitude;
        }

        /// <summary>
        /// Create Hann window.
        /// </summary>
        private float[] CreateHannWindow(int length)
        {
            float[] window = new float[length];
            for (int i = 0; i < length; i++)
            {
                window[i] = 0.5f * (1 - (float)Math.Cos(2 * Math.PI * i / (length - 1)));
            }
            return window;
        }

        /// <summary>
        /// Create mel filterbank matrix.
        /// </summary>
        private float[,] CreateMelFilterbank()
        {
            int numBins = FFTSize / 2 + 1;
            float[,] filterbank = new float[MelBins, numBins];

            // Convert Hz to Mel
            float melMin = HzToMel(FMin);
            float melMax = HzToMel(FMax);

            // Create mel points
            float[] melPoints = new float[MelBins + 2];
            for (int i = 0; i < MelBins + 2; i++)
            {
                melPoints[i] = melMin + (melMax - melMin) * i / (MelBins + 1);
            }

            // Convert back to Hz
            float[] hzPoints = new float[MelBins + 2];
            for (int i = 0; i < MelBins + 2; i++)
            {
                hzPoints[i] = MelToHz(melPoints[i]);
            }

            // Convert to FFT bin indices
            int[] binPoints = new int[MelBins + 2];
            for (int i = 0; i < MelBins + 2; i++)
            {
                binPoints[i] = (int)Math.Floor((FFTSize + 1) * hzPoints[i] / SampleRate);
            }

            // Create triangular filters
            for (int m = 0; m < MelBins; m++)
            {
                int start = binPoints[m];
                int center = binPoints[m + 1];
                int end = binPoints[m + 2];

                for (int f = start; f < center && f < numBins; f++)
                {
                    if (center != start)
                        filterbank[m, f] = (float)(f - start) / (center - start);
                }
                for (int f = center; f < end && f < numBins; f++)
                {
                    if (end != center)
                        filterbank[m, f] = (float)(end - f) / (end - center);
                }
            }

            return filterbank;
        }

        /// <summary>
        /// Convert frequency in Hz to Mel scale.
        /// </summary>
        private float HzToMel(float hz)
        {
            return 2595f * (float)Math.Log10(1 + hz / 700f);
        }

        /// <summary>
        /// Convert frequency in Mel scale to Hz.
        /// </summary>
        private float MelToHz(float mel)
        {
            return 700f * ((float)Math.Pow(10, mel / 2595f) - 1);
        }

        /// <summary>
        /// In-place Cooley-Tukey FFT.
        /// </summary>
        private void FFT(Complex[] data)
        {
            int n = data.Length;
            if (n <= 1) return;

            // Bit-reversal permutation
            int bits = (int)Math.Log2(n);
            for (int i = 0; i < n; i++)
            {
                int j = BitReverse(i, bits);
                if (j > i)
                {
                    (data[i], data[j]) = (data[j], data[i]);
                }
            }

            // Cooley-Tukey iterative FFT
            for (int len = 2; len <= n; len *= 2)
            {
                double angle = -2 * Math.PI / len;
                Complex wlen = new Complex(Math.Cos(angle), Math.Sin(angle));

                for (int i = 0; i < n; i += len)
                {
                    Complex w = Complex.One;
                    for (int j = 0; j < len / 2; j++)
                    {
                        Complex u = data[i + j];
                        Complex t = w * data[i + j + len / 2];
                        data[i + j] = u + t;
                        data[i + j + len / 2] = u - t;
                        w *= wlen;
                    }
                }
            }
        }

        /// <summary>
        /// In-place inverse FFT.
        /// </summary>
        private void IFFT(Complex[] data)
        {
            int n = data.Length;

            // Conjugate
            for (int i = 0; i < n; i++)
            {
                data[i] = Complex.Conjugate(data[i]);
            }

            // Forward FFT
            FFT(data);

            // Conjugate and scale
            for (int i = 0; i < n; i++)
            {
                data[i] = Complex.Conjugate(data[i]) / n;
            }
        }

        /// <summary>
        /// Bit reversal for FFT.
        /// </summary>
        private int BitReverse(int x, int bits)
        {
            int result = 0;
            for (int i = 0; i < bits; i++)
            {
                result = (result << 1) | (x & 1);
                x >>= 1;
            }
            return result;
        }

        /// <summary>
        /// Normalize mel spectrogram to range [0, 1].
        /// </summary>
        public float[,] NormalizeMelSpectrogram(float[,] melSpec)
        {
            int rows = melSpec.GetLength(0);
            int cols = melSpec.GetLength(1);

            float min = float.MaxValue;
            float max = float.MinValue;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (melSpec[i, j] < min) min = melSpec[i, j];
                    if (melSpec[i, j] > max) max = melSpec[i, j];
                }
            }

            float range = max - min;
            if (range < 1e-6f) range = 1f;

            float[,] normalized = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    normalized[i, j] = (melSpec[i, j] - min) / range;
                }
            }

            return normalized;
        }

        /// <summary>
        /// Denormalize mel spectrogram from [0, 1] range.
        /// </summary>
        public float[,] DenormalizeMelSpectrogram(float[,] normalized, float min, float max)
        {
            int rows = normalized.GetLength(0);
            int cols = normalized.GetLength(1);

            float[,] melSpec = new float[rows, cols];
            float range = max - min;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    melSpec[i, j] = normalized[i, j] * range + min;
                }
            }

            return melSpec;
        }
    }
}
