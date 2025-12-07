using System;
using System.IO;
using NeuralNetwork.Models.TTS;
using NeuralNetwork.Processing.Audio;

namespace NeuralNetwork.Inference.Audio
{
    /// <summary>
    /// Speech synthesizer for text-to-speech inference.
    /// Supports both CPU (TTSModel) and GPU (TTSModelCuda) models.
    /// </summary>
    public class SpeechSynthesizer : IDisposable
    {
        // Use dynamic to support both model types
        private readonly dynamic _model;
        private readonly bool _isCudaModel;

        public TTSConfig Config { get; }
        public MelSpectrogram MelProcessor { get; }
        public AudioPreProcessing AudioProcessor { get; }

        /// <summary>
        /// Create synthesizer with CPU model.
        /// </summary>
        public SpeechSynthesizer(TTSModel model)
        {
            _model = model;
            _isCudaModel = false;
            Config = model.Config;

            MelProcessor = new MelSpectrogram(
                sampleRate: Config.SampleRate,
                fftSize: Config.FFTSize,
                hopLength: Config.HopLength,
                melBins: Config.MelBins
            );

            AudioProcessor = new AudioPreProcessing
            {
                TargetSampleRate = Config.SampleRate
            };

            model.SetTraining(false);
        }

        /// <summary>
        /// Create synthesizer with CUDA model for GPU-accelerated synthesis.
        /// </summary>
        public SpeechSynthesizer(TTSModelCuda model)
        {
            _model = model;
            _isCudaModel = true;
            Config = model.Config;

            MelProcessor = new MelSpectrogram(
                sampleRate: Config.SampleRate,
                fftSize: Config.FFTSize,
                hopLength: Config.HopLength,
                melBins: Config.MelBins
            );

            AudioProcessor = new AudioPreProcessing
            {
                TargetSampleRate = Config.SampleRate
            };

            model.SetTraining(false);
        }

        /// <summary>
        /// Synthesize speech from text.
        /// </summary>
        /// <param name="text">Input text to synthesize</param>
        /// <param name="speakerId">Speaker ID for multi-speaker models (default 0)</param>
        /// <returns>Audio waveform samples</returns>
        public float[] Synthesize(string text, int speakerId = 0)
        {
            // Convert text to tokens
            int[] tokens = _model.TextToTokens(text);

            // Run TTS model to get mel spectrogram
            var result = _model.Forward(tokens, null, speakerId);
            float[,] melOutput = result.Item1;
            float[] stopTokens = result.Item2;
            float[,] attentionWeights = result.Item3;

            // Convert mel spectrogram to waveform using Griffin-Lim
            float[] waveform = MelProcessor.MelSpectrogramToWaveform(melOutput, iterations: 60);

            // Remove pre-emphasis if applied
            waveform = AudioProcessor.RemovePreEmphasis(waveform);

            // Normalize output
            waveform = AudioProcessor.NormalizeAudio(waveform);

            return waveform;
        }

        /// <summary>
        /// Synthesize speech and save to WAV file.
        /// </summary>
        public void SynthesizeToFile(string text, string outputPath, int speakerId = 0)
        {
            float[] waveform = Synthesize(text, speakerId);
            AudioProcessor.SaveWav(outputPath, waveform, Config.SampleRate);
        }

        /// <summary>
        /// Synthesize speech with detailed output.
        /// </summary>
        public SynthesisResult SynthesizeDetailed(string text, int speakerId = 0)
        {
            int[] tokens = _model.TextToTokens(text);

            var result = _model.Forward(tokens, null, speakerId);
            float[,] melOutput = result.Item1;
            float[] stopTokens = result.Item2;
            float[,] attentionWeights = result.Item3;

            float[] waveform = MelProcessor.MelSpectrogramToWaveform(melOutput, iterations: 60);
            waveform = AudioProcessor.RemovePreEmphasis(waveform);
            waveform = AudioProcessor.NormalizeAudio(waveform);

            return new SynthesisResult
            {
                Text = text,
                Tokens = tokens,
                MelSpectrogram = melOutput,
                StopTokens = stopTokens,
                AttentionWeights = attentionWeights,
                Waveform = waveform,
                SampleRate = Config.SampleRate,
                DurationSeconds = (float)waveform.Length / Config.SampleRate
            };
        }

        /// <summary>
        /// Get mel spectrogram without vocoder (for debugging).
        /// </summary>
        public float[,] GetMelSpectrogram(string text, int speakerId = 0)
        {
            int[] tokens = _model.TextToTokens(text);
            var result = _model.Forward(tokens, null, speakerId);
            return (float[,])result.Item1;
        }

        /// <summary>
        /// Clone a voice from reference audio.
        /// </summary>
        /// <param name="referenceAudioPath">Path to reference WAV file</param>
        /// <param name="text">Text to synthesize in cloned voice</param>
        /// <returns>Audio waveform in cloned voice</returns>
        public float[] CloneVoice(string referenceAudioPath, string text)
        {
            // Load reference audio
            float[] referenceAudio = AudioProcessor.LoadWav(referenceAudioPath);
            referenceAudio = AudioProcessor.ApplyPreEmphasis(referenceAudio);

            // Get mel spectrogram of reference
            float[,] referenceMel = MelProcessor.WaveformToMelSpectrogram(referenceAudio);

            // For voice cloning, we would need a speaker encoder
            // For now, use default speaker
            // TODO: Implement speaker encoder for voice cloning
            Console.WriteLine("Note: Full voice cloning requires a speaker encoder. Using default voice.");

            return Synthesize(text, speakerId: 0);
        }

        /// <summary>
        /// Process streaming audio input (for speech-to-speech).
        /// </summary>
        public float[] ProcessStreamingInput(float[] inputAudio, string targetText = null)
        {
            // Convert input audio to mel spectrogram
            float[] processed = AudioProcessor.ApplyPreEmphasis(inputAudio);
            float[,] inputMel = MelProcessor.WaveformToMelSpectrogram(processed);

            // If no target text, we would need ASR to transcribe
            // For now, this is a placeholder for speech-to-speech
            if (targetText == null)
            {
                Console.WriteLine("Note: ASR not implemented. Provide target text for synthesis.");
                return Array.Empty<float>();
            }

            // Synthesize with the target text
            return Synthesize(targetText);
        }

        /// <summary>
        /// Interactive synthesis with real-time microphone input.
        /// </summary>
        public void InteractiveMode()
        {
            Console.WriteLine("Speech Synthesizer - Interactive Mode");
            Console.WriteLine("=====================================");
            Console.WriteLine("Commands:");
            Console.WriteLine("  speak <text>  - Synthesize and play text");
            Console.WriteLine("  save <file> <text> - Save synthesized audio to file");
            Console.WriteLine("  mic           - Record from microphone");
            Console.WriteLine("  quit          - Exit interactive mode");
            Console.WriteLine();

            while (true)
            {
                Console.Write("> ");
                string input = Console.ReadLine()?.Trim();

                if (string.IsNullOrEmpty(input)) continue;

                if (input.Equals("quit", StringComparison.OrdinalIgnoreCase) ||
                    input.Equals("exit", StringComparison.OrdinalIgnoreCase))
                {
                    break;
                }

                if (input.StartsWith("speak ", StringComparison.OrdinalIgnoreCase))
                {
                    string text = input.Substring(6).Trim();
                    try
                    {
                        Console.WriteLine($"Synthesizing: \"{text}\"");
                        var result = SynthesizeDetailed(text);
                        Console.WriteLine($"Generated {result.DurationSeconds:F2}s of audio ({result.Waveform.Length} samples)");
                        Console.WriteLine($"Mel frames: {result.MelSpectrogram.GetLength(0)}");

                        // Save to temp file
                        string tempPath = Path.Combine(Path.GetTempPath(), "tts_output.wav");
                        AudioProcessor.SaveWav(tempPath, result.Waveform, Config.SampleRate);
                        Console.WriteLine($"Saved to: {tempPath}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error: {ex.Message}");
                    }
                }
                else if (input.StartsWith("save ", StringComparison.OrdinalIgnoreCase))
                {
                    var parts = input.Substring(5).Trim().Split(' ', 2);
                    if (parts.Length < 2)
                    {
                        Console.WriteLine("Usage: save <filename> <text>");
                        continue;
                    }

                    string filename = parts[0];
                    string text = parts[1];

                    try
                    {
                        SynthesizeToFile(text, filename);
                        Console.WriteLine($"Saved to: {filename}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error: {ex.Message}");
                    }
                }
                else if (input.Equals("mic", StringComparison.OrdinalIgnoreCase))
                {
                    if (!MicrophoneStream.IsMicrophoneAvailable())
                    {
                        Console.WriteLine("No microphone available.");
                        continue;
                    }

                    Console.WriteLine("Recording... (press Enter to stop)");

                    using var mic = new MicrophoneStream(sampleRate: Config.SampleRate);
                    mic.Start();

                    // Wait for Enter key
                    Console.ReadLine();
                    mic.Stop();

                    float[] audio = mic.GetAllQueuedAudio();
                    Console.WriteLine($"Recorded {audio.Length} samples ({(float)audio.Length / Config.SampleRate:F2}s)");

                    // Save recording
                    string tempPath = Path.Combine(Path.GetTempPath(), "mic_recording.wav");
                    AudioProcessor.SaveWav(tempPath, audio, Config.SampleRate);
                    Console.WriteLine($"Saved to: {tempPath}");
                }
                else
                {
                    Console.WriteLine("Unknown command. Type 'speak <text>' to synthesize.");
                }
            }
        }

        public void Dispose()
        {
            if (_isCudaModel && _model is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
    }

    /// <summary>
    /// Detailed synthesis result.
    /// </summary>
    public class SynthesisResult
    {
        public string Text { get; set; }
        public int[] Tokens { get; set; }
        public float[,] MelSpectrogram { get; set; }
        public float[] StopTokens { get; set; }
        public float[,] AttentionWeights { get; set; }
        public float[] Waveform { get; set; }
        public int SampleRate { get; set; }
        public float DurationSeconds { get; set; }
    }
}
