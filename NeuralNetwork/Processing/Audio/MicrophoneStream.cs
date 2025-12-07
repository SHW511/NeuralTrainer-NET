using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork.Processing.Audio
{
    /// <summary>
    /// Real-time microphone audio capture for streaming input.
    /// Uses Windows Multimedia API (winmm.dll).
    /// </summary>
    public class MicrophoneStream : IDisposable
    {
        // Windows Multimedia API imports
        [DllImport("winmm.dll", SetLastError = true)]
        private static extern int waveInGetNumDevs();

        [DllImport("winmm.dll", SetLastError = true)]
        private static extern int waveInOpen(out IntPtr phwi, uint uDeviceID, ref WAVEFORMATEX lpFormat,
            WaveInProc dwCallback, IntPtr dwInstance, uint fdwOpen);

        [DllImport("winmm.dll", SetLastError = true)]
        private static extern int waveInClose(IntPtr hwi);

        [DllImport("winmm.dll", SetLastError = true)]
        private static extern int waveInPrepareHeader(IntPtr hwi, IntPtr lpWaveHdr, uint uSize);

        [DllImport("winmm.dll", SetLastError = true)]
        private static extern int waveInUnprepareHeader(IntPtr hwi, IntPtr lpWaveHdr, uint uSize);

        [DllImport("winmm.dll", SetLastError = true)]
        private static extern int waveInAddBuffer(IntPtr hwi, IntPtr lpWaveHdr, uint uSize);

        [DllImport("winmm.dll", SetLastError = true)]
        private static extern int waveInStart(IntPtr hwi);

        [DllImport("winmm.dll", SetLastError = true)]
        private static extern int waveInStop(IntPtr hwi);

        [DllImport("winmm.dll", SetLastError = true)]
        private static extern int waveInReset(IntPtr hwi);

        private delegate void WaveInProc(IntPtr hwi, uint uMsg, IntPtr dwInstance, IntPtr dwParam1, IntPtr dwParam2);

        private const uint CALLBACK_FUNCTION = 0x00030000;
        private const uint WAVE_MAPPER = unchecked((uint)-1);
        private const uint WIM_DATA = 0x3C4;

        [StructLayout(LayoutKind.Sequential)]
        private struct WAVEFORMATEX
        {
            public ushort wFormatTag;
            public ushort nChannels;
            public uint nSamplesPerSec;
            public uint nAvgBytesPerSec;
            public ushort nBlockAlign;
            public ushort wBitsPerSample;
            public ushort cbSize;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct WAVEHDR
        {
            public IntPtr lpData;
            public uint dwBufferLength;
            public uint dwBytesRecorded;
            public IntPtr dwUser;
            public uint dwFlags;
            public uint dwLoops;
            public IntPtr lpNext;
            public IntPtr reserved;
        }

        // Configuration
        public int SampleRate { get; }
        public int Channels { get; }
        public int BitsPerSample { get; }
        public int BufferSizeMs { get; }

        // State
        private IntPtr _waveIn;
        private IntPtr[] _bufferPtrs;
        private IntPtr[] _headerPtrs;
        private GCHandle[] _bufferHandles;
        private WaveInProc _callback;
        private bool _isRecording;
        private bool _disposed;

        // Audio buffer
        private ConcurrentQueue<float[]> _audioQueue;
        private int _bufferCount = 4;

        // Events
        public event Action<float[]> OnAudioReceived;

        public MicrophoneStream(int sampleRate = 16000, int channels = 1, int bitsPerSample = 16, int bufferSizeMs = 100)
        {
            SampleRate = sampleRate;
            Channels = channels;
            BitsPerSample = bitsPerSample;
            BufferSizeMs = bufferSizeMs;

            _audioQueue = new ConcurrentQueue<float[]>();
        }

        /// <summary>
        /// Check if microphone is available.
        /// </summary>
        public static bool IsMicrophoneAvailable()
        {
            try
            {
                return waveInGetNumDevs() > 0;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Start recording from microphone.
        /// </summary>
        public void Start()
        {
            if (_isRecording) return;

            // Setup wave format
            var format = new WAVEFORMATEX
            {
                wFormatTag = 1, // PCM
                nChannels = (ushort)Channels,
                nSamplesPerSec = (uint)SampleRate,
                wBitsPerSample = (ushort)BitsPerSample,
                nBlockAlign = (ushort)(Channels * BitsPerSample / 8),
                nAvgBytesPerSec = (uint)(SampleRate * Channels * BitsPerSample / 8),
                cbSize = 0
            };

            _callback = WaveInCallback;

            int result = waveInOpen(out _waveIn, WAVE_MAPPER, ref format, _callback, IntPtr.Zero, CALLBACK_FUNCTION);
            if (result != 0)
            {
                throw new InvalidOperationException($"Failed to open microphone: error code {result}");
            }

            // Allocate buffers
            int bufferSize = SampleRate * Channels * (BitsPerSample / 8) * BufferSizeMs / 1000;
            _bufferPtrs = new IntPtr[_bufferCount];
            _headerPtrs = new IntPtr[_bufferCount];
            _bufferHandles = new GCHandle[_bufferCount];

            for (int i = 0; i < _bufferCount; i++)
            {
                byte[] buffer = new byte[bufferSize];
                _bufferHandles[i] = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                _bufferPtrs[i] = _bufferHandles[i].AddrOfPinnedObject();

                var header = new WAVEHDR
                {
                    lpData = _bufferPtrs[i],
                    dwBufferLength = (uint)bufferSize,
                    dwFlags = 0
                };

                _headerPtrs[i] = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(WAVEHDR)));
                Marshal.StructureToPtr(header, _headerPtrs[i], false);

                waveInPrepareHeader(_waveIn, _headerPtrs[i], (uint)Marshal.SizeOf(typeof(WAVEHDR)));
                waveInAddBuffer(_waveIn, _headerPtrs[i], (uint)Marshal.SizeOf(typeof(WAVEHDR)));
            }

            result = waveInStart(_waveIn);
            if (result != 0)
            {
                throw new InvalidOperationException($"Failed to start recording: error code {result}");
            }

            _isRecording = true;
        }

        /// <summary>
        /// Stop recording.
        /// </summary>
        public void Stop()
        {
            if (!_isRecording) return;

            _isRecording = false;

            waveInStop(_waveIn);
            waveInReset(_waveIn);

            // Clean up buffers
            for (int i = 0; i < _bufferCount; i++)
            {
                if (_headerPtrs[i] != IntPtr.Zero)
                {
                    waveInUnprepareHeader(_waveIn, _headerPtrs[i], (uint)Marshal.SizeOf(typeof(WAVEHDR)));
                    Marshal.FreeHGlobal(_headerPtrs[i]);
                }
                if (_bufferHandles[i].IsAllocated)
                {
                    _bufferHandles[i].Free();
                }
            }

            waveInClose(_waveIn);
            _waveIn = IntPtr.Zero;
        }

        private void WaveInCallback(IntPtr hwi, uint uMsg, IntPtr dwInstance, IntPtr dwParam1, IntPtr dwParam2)
        {
            if (uMsg == WIM_DATA && _isRecording)
            {
                var header = Marshal.PtrToStructure<WAVEHDR>(dwParam1);

                // Convert to float samples
                int sampleCount = (int)(header.dwBytesRecorded / (BitsPerSample / 8));
                float[] samples = new float[sampleCount];

                // Copy data from unmanaged memory
                byte[] rawData = new byte[header.dwBytesRecorded];
                Marshal.Copy(header.lpData, rawData, 0, (int)header.dwBytesRecorded);

                if (BitsPerSample == 16)
                {
                    for (int i = 0; i < sampleCount; i++)
                    {
                        short sample = (short)(rawData[i * 2] | (rawData[i * 2 + 1] << 8));
                        samples[i] = sample / 32768f;
                    }
                }
                else if (BitsPerSample == 8)
                {
                    for (int i = 0; i < sampleCount; i++)
                    {
                        samples[i] = (rawData[i] - 128) / 128f;
                    }
                }

                // Add to queue and raise event
                _audioQueue.Enqueue(samples);
                OnAudioReceived?.Invoke(samples);

                // Re-add buffer for more recording
                if (_isRecording)
                {
                    waveInAddBuffer(hwi, dwParam1, (uint)Marshal.SizeOf(typeof(WAVEHDR)));
                }
            }
        }

        /// <summary>
        /// Get queued audio data.
        /// </summary>
        public bool TryGetAudio(out float[] audio)
        {
            return _audioQueue.TryDequeue(out audio);
        }

        /// <summary>
        /// Get all queued audio concatenated.
        /// </summary>
        public float[] GetAllQueuedAudio()
        {
            var allAudio = new System.Collections.Generic.List<float>();

            while (_audioQueue.TryDequeue(out float[] chunk))
            {
                allAudio.AddRange(chunk);
            }

            return allAudio.ToArray();
        }

        /// <summary>
        /// Clear audio queue.
        /// </summary>
        public void ClearQueue()
        {
            while (_audioQueue.TryDequeue(out _)) { }
        }

        /// <summary>
        /// Record audio for a specified duration.
        /// </summary>
        public async Task<float[]> RecordAsync(int durationMs, CancellationToken cancellationToken = default)
        {
            var audioData = new System.Collections.Generic.List<float>();

            Start();

            try
            {
                int totalSamples = SampleRate * durationMs / 1000;

                while (audioData.Count < totalSamples && !cancellationToken.IsCancellationRequested)
                {
                    if (TryGetAudio(out float[] chunk))
                    {
                        audioData.AddRange(chunk);
                    }
                    else
                    {
                        await Task.Delay(10, cancellationToken);
                    }
                }
            }
            finally
            {
                Stop();
            }

            return audioData.ToArray();
        }

        /// <summary>
        /// Record until silence is detected.
        /// </summary>
        public async Task<float[]> RecordUntilSilenceAsync(
            float silenceThreshold = 0.01f,
            int silenceDurationMs = 1000,
            int maxDurationMs = 30000,
            CancellationToken cancellationToken = default)
        {
            var audioData = new System.Collections.Generic.List<float>();
            int silenceSamples = 0;
            int silenceSamplesThreshold = SampleRate * silenceDurationMs / 1000;
            int maxSamples = SampleRate * maxDurationMs / 1000;

            Start();

            try
            {
                while (audioData.Count < maxSamples && !cancellationToken.IsCancellationRequested)
                {
                    if (TryGetAudio(out float[] chunk))
                    {
                        audioData.AddRange(chunk);

                        // Check for silence
                        float maxAbs = 0;
                        foreach (float sample in chunk)
                        {
                            float abs = Math.Abs(sample);
                            if (abs > maxAbs) maxAbs = abs;
                        }

                        if (maxAbs < silenceThreshold)
                        {
                            silenceSamples += chunk.Length;
                            if (silenceSamples >= silenceSamplesThreshold && audioData.Count > SampleRate / 2)
                            {
                                break; // Silence detected
                            }
                        }
                        else
                        {
                            silenceSamples = 0;
                        }
                    }
                    else
                    {
                        await Task.Delay(10, cancellationToken);
                    }
                }
            }
            finally
            {
                Stop();
            }

            return audioData.ToArray();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            Stop();
            GC.SuppressFinalize(this);
        }

        ~MicrophoneStream()
        {
            Dispose();
        }
    }
}
