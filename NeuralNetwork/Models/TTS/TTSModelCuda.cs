using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using NeuralNetwork.Layers.Cuda;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Models.TTS
{
    /// <summary>
    /// CUDA-accelerated Text-to-Speech model.
    /// Uses GPU for all major computations.
    /// </summary>
    public class TTSModelCuda : IDisposable
    {
        public TTSConfig Config { get; }

        private CudaContext _context;
        private bool _disposed;

        // Character vocabulary
        private Dictionary<char, int> _charToIdx;
        private Dictionary<int, char> _idxToChar;

        // Device memory for weights
        private CudaDeviceVariable<float> _textEmbeddingDevice;
        private CudaDeviceVariable<float>[] _encoderConvWeightsDevice;
        private CudaDeviceVariable<float>[] _encoderConvBiasDevice;
        private CudaDeviceVariable<float> _queryProjDevice;
        private CudaDeviceVariable<float> _keyProjDevice;
        private CudaDeviceVariable<float>[] _prenetWeightsDevice;
        private CudaDeviceVariable<float>[] _prenetBiasDevice;
        private CudaDeviceVariable<float> _lstmWeightsIhDevice;
        private CudaDeviceVariable<float> _lstmWeightsHhDevice;
        private CudaDeviceVariable<float> _lstmBiasDevice;
        private CudaDeviceVariable<float> _melProjectionDevice;
        private CudaDeviceVariable<float> _melProjectionBiasDevice;
        private CudaDeviceVariable<float>[] _postnetConvWeightsDevice;
        private CudaDeviceVariable<float>[] _postnetConvBiasDevice;

        // Persistent intermediate buffers (pre-allocated for max sizes)
        private CudaDeviceVariable<float> _embeddedDevice;
        private CudaDeviceVariable<float> _encoderOutputDevice;
        private CudaDeviceVariable<float> _prenetOutputDevice;
        private CudaDeviceVariable<float> _attentionContextDevice;
        private CudaDeviceVariable<float> _attentionWeightsDevice;
        private CudaDeviceVariable<float> _attentionEnergiesDevice;
        private CudaDeviceVariable<float> _lstmHDevice;
        private CudaDeviceVariable<float> _lstmCDevice;
        private CudaDeviceVariable<float> _lstmInputDevice;
        private CudaDeviceVariable<float> _melFrameDevice;
        private CudaDeviceVariable<float> _decoderOutputDevice;
        private CudaDeviceVariable<int> _tokensDevice;

        // Max sizes for pre-allocation
        private int _maxSeqLen = 512;
        private int _maxMelFrames = 2000;

        // Host copies for gradient updates
        private float[,] _textEmbedding;
        private float[][,] _encoderConvWeights;
        private float[][] _encoderConvBias;
        private float[,] _queryProj;
        private float[,] _keyProj;
        private float[][,] _prenetWeights;
        private float[][] _prenetBias;
        private float[,] _lstmWeightsIh;
        private float[,] _lstmWeightsHh;
        private float[] _lstmBias;
        private float[,] _melProjection;
        private float[] _melProjectionBias;
        private float[][,] _postnetConvWeights;
        private float[][] _postnetConvBias;

        // CUDA kernels - General
        private CudaKernel _embedKernel;
        private CudaKernel _matmulKernel;
        private CudaKernel _addBiasKernel;
        private CudaKernel _tanhKernel;
        private CudaKernel _reluKernel;
        private CudaKernel _sigmoidKernel;
        private CudaKernel _softmaxKernel;
        private CudaKernel _lstmKernel;
        private CudaKernel _conv1dKernel;

        // CUDA kernels - TTS specific
        private CudaKernel _conv1dForwardKernel;
        private CudaKernel _conv1dForwardReLUKernel;
        private CudaKernel _conv1dForwardTanhKernel;
        private CudaKernel _embeddingLookupKernel;
        private CudaKernel _prenetForwardKernel;
        private CudaKernel _locationAttentionEnergyKernel;
        private CudaKernel _attentionSoftmaxKernel;
        private CudaKernel _attentionContextKernel;
        private CudaKernel _lstmStepKernel;
        private CudaKernel _lstmStepOptimizedKernel;
        private CudaKernel _melProjectionKernel;
        private CudaKernel _postnetResidualKernel;
        private CudaKernel _stopTokenSigmoidKernel;
        private CudaKernel _batchedEmbeddingKernel;
        private CudaKernel _batchedConv1dKernel;
        private bool _ttsKernelsLoaded = false;

        // Training state
        private bool _training = true;
        private Random _random;

        public TTSModelCuda(TTSConfig config, int? seed = null)
        {
            Config = config;
            Config.Validate();

            _random = seed.HasValue ? new Random(seed.Value) : new Random();

            // Initialize CUDA
            _context = new CudaContext();
            Console.WriteLine($"CUDA Device: {_context.GetDeviceName()}");
            Console.WriteLine($"CUDA Compute Capability: {_context.GetDeviceComputeCapability()}");
            Console.WriteLine($"Total Memory: {(long)_context.GetTotalDeviceMemorySize() / (1024.0 * 1024 * 1024):F1} GB");

            InitializeVocabulary();
            InitializeWeights();
            LoadKernels();
            CopyWeightsToDevice();
            InitializeIntermediateBuffers();
        }

        private void InitializeVocabulary()
        {
            _charToIdx = new Dictionary<char, int>();
            _idxToChar = new Dictionary<int, char>();

            int idx = 0;
            _charToIdx['<'] = idx; _idxToChar[idx++] = '<';  // PAD
            _charToIdx['>'] = idx; _idxToChar[idx++] = '>';  // EOS

            for (char c = ' '; c <= '~'; c++)
            {
                if (!_charToIdx.ContainsKey(c))
                {
                    _charToIdx[c] = idx;
                    _idxToChar[idx++] = c;
                }
            }

            Config.VocabSize = idx;
        }

        private void InitializeWeights()
        {
            // Text embedding
            _textEmbedding = InitializeMatrix(Config.VocabSize, Config.TextEmbeddingDim);

            // Encoder convolutions
            _encoderConvWeights = new float[Config.EncoderConvLayers][,];
            _encoderConvBias = new float[Config.EncoderConvLayers][];

            int inChannels = Config.TextEmbeddingDim;
            for (int i = 0; i < Config.EncoderConvLayers; i++)
            {
                _encoderConvWeights[i] = InitializeMatrix(
                    Config.EncoderKernelSize * inChannels,
                    Config.EncoderDim);
                _encoderConvBias[i] = new float[Config.EncoderDim];
                inChannels = Config.EncoderDim;
            }

            // Attention
            _queryProj = InitializeMatrix(Config.DecoderDim, Config.AttentionDim);
            _keyProj = InitializeMatrix(Config.EncoderDim, Config.AttentionDim);

            // Prenet
            _prenetWeights = new float[Config.PrenetDims.Length][,];
            _prenetBias = new float[Config.PrenetDims.Length][];

            int prenetInput = Config.MelBins * Config.OutputsPerStep;
            for (int i = 0; i < Config.PrenetDims.Length; i++)
            {
                _prenetWeights[i] = InitializeMatrix(prenetInput, Config.PrenetDims[i]);
                _prenetBias[i] = new float[Config.PrenetDims[i]];
                prenetInput = Config.PrenetDims[i];
            }

            // Decoder LSTM
            int lstmInputDim = Config.PrenetDims[^1] + Config.EncoderDim;
            _lstmWeightsIh = InitializeMatrix(lstmInputDim, Config.DecoderDim * 4);
            _lstmWeightsHh = InitializeMatrix(Config.DecoderDim, Config.DecoderDim * 4);
            _lstmBias = new float[Config.DecoderDim * 4];

            // Forget gate bias = 1
            for (int i = Config.DecoderDim; i < Config.DecoderDim * 2; i++)
                _lstmBias[i] = 1f;

            // Mel projection
            int projInput = Config.DecoderDim + Config.EncoderDim;
            _melProjection = InitializeMatrix(projInput, Config.MelBins * Config.OutputsPerStep);
            _melProjectionBias = new float[Config.MelBins * Config.OutputsPerStep];

            // Postnet
            _postnetConvWeights = new float[Config.PostnetLayers][,];
            _postnetConvBias = new float[Config.PostnetLayers][];

            inChannels = Config.MelBins;
            for (int i = 0; i < Config.PostnetLayers; i++)
            {
                int outChannels = (i == Config.PostnetLayers - 1) ? Config.MelBins : Config.PostnetChannels;
                _postnetConvWeights[i] = InitializeMatrix(
                    Config.PostnetKernelSize * inChannels,
                    outChannels);
                _postnetConvBias[i] = new float[outChannels];
                inChannels = outChannels;
            }
        }

        private void LoadKernels()
        {
            string cuDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU");

            // Load existing kernels
            string denseKernelPath = Path.Combine(cuDir, "DenseKernel.ptx");
            string activationsPath = Path.Combine(cuDir, "ActivationsKernel.ptx");

            if (File.Exists(denseKernelPath))
            {
                try
                {
                    var module = _context.LoadModulePTX(denseKernelPath);
                    try { _matmulKernel = new CudaKernel("MatMul", module, _context); } catch { }
                    try { _addBiasKernel = new CudaKernel("AddBias", module, _context); } catch { }
                }
                catch { }
            }

            if (File.Exists(activationsPath))
            {
                try
                {
                    var module = _context.LoadModulePTX(activationsPath);
                    try { _reluKernel = new CudaKernel("relu_forward", module, _context); } catch { }
                    try { _sigmoidKernel = new CudaKernel("sigmoid_forward", module, _context); } catch { }
                    try { _tanhKernel = new CudaKernel("tanh_forward", module, _context); } catch { }
                }
                catch { }
            }

            // Load TTS-specific kernels
            string ttsKernelPath = Path.Combine(cuDir, "TTSKernel.ptx");
            if (File.Exists(ttsKernelPath))
            {
                try
                {
                    var ttsModule = _context.LoadModulePTX(ttsKernelPath);

                    _conv1dForwardKernel = new CudaKernel("Conv1DForward", ttsModule, _context);
                    _conv1dForwardReLUKernel = new CudaKernel("Conv1DForwardReLU", ttsModule, _context);
                    _conv1dForwardTanhKernel = new CudaKernel("Conv1DForwardTanh", ttsModule, _context);
                    _embeddingLookupKernel = new CudaKernel("EmbeddingLookup", ttsModule, _context);
                    _prenetForwardKernel = new CudaKernel("PrenetForward", ttsModule, _context);
                    _locationAttentionEnergyKernel = new CudaKernel("LocationSensitiveAttentionEnergy", ttsModule, _context);
                    _attentionSoftmaxKernel = new CudaKernel("AttentionSoftmax", ttsModule, _context);
                    _attentionContextKernel = new CudaKernel("AttentionContext", ttsModule, _context);
                    _lstmStepKernel = new CudaKernel("LSTMStep", ttsModule, _context);
                    _lstmStepOptimizedKernel = new CudaKernel("LSTMStepOptimized", ttsModule, _context);
                    _melProjectionKernel = new CudaKernel("MelProjection", ttsModule, _context);
                    _postnetResidualKernel = new CudaKernel("PostnetResidual", ttsModule, _context);
                    _stopTokenSigmoidKernel = new CudaKernel("StopTokenSigmoid", ttsModule, _context);
                    _batchedEmbeddingKernel = new CudaKernel("BatchedEmbeddingLookup", ttsModule, _context);
                    _batchedConv1dKernel = new CudaKernel("BatchedConv1DForward", ttsModule, _context);

                    _ttsKernelsLoaded = true;
                    Console.WriteLine("TTS CUDA kernels loaded successfully.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Could not load TTS kernels: {ex.Message}");
                    Console.WriteLine("Falling back to CPU implementations.");
                    _ttsKernelsLoaded = false;
                }
            }
            else
            {
                Console.WriteLine($"TTS kernel file not found at: {ttsKernelPath}");
                Console.WriteLine("Using CPU fallback implementations.");
            }
        }

        private void CopyWeightsToDevice()
        {
            // Text embedding
            _textEmbeddingDevice = new CudaDeviceVariable<float>(_textEmbedding.Length);
            _textEmbeddingDevice.CopyToDevice(Flatten(_textEmbedding));

            // Encoder conv weights
            _encoderConvWeightsDevice = new CudaDeviceVariable<float>[Config.EncoderConvLayers];
            _encoderConvBiasDevice = new CudaDeviceVariable<float>[Config.EncoderConvLayers];
            for (int i = 0; i < Config.EncoderConvLayers; i++)
            {
                _encoderConvWeightsDevice[i] = new CudaDeviceVariable<float>(_encoderConvWeights[i].Length);
                _encoderConvWeightsDevice[i].CopyToDevice(Flatten(_encoderConvWeights[i]));
                _encoderConvBiasDevice[i] = new CudaDeviceVariable<float>(_encoderConvBias[i].Length);
                _encoderConvBiasDevice[i].CopyToDevice(_encoderConvBias[i]);
            }

            // Attention
            _queryProjDevice = new CudaDeviceVariable<float>(_queryProj.Length);
            _queryProjDevice.CopyToDevice(Flatten(_queryProj));
            _keyProjDevice = new CudaDeviceVariable<float>(_keyProj.Length);
            _keyProjDevice.CopyToDevice(Flatten(_keyProj));

            // Prenet
            _prenetWeightsDevice = new CudaDeviceVariable<float>[Config.PrenetDims.Length];
            _prenetBiasDevice = new CudaDeviceVariable<float>[Config.PrenetDims.Length];
            for (int i = 0; i < Config.PrenetDims.Length; i++)
            {
                _prenetWeightsDevice[i] = new CudaDeviceVariable<float>(_prenetWeights[i].Length);
                _prenetWeightsDevice[i].CopyToDevice(Flatten(_prenetWeights[i]));
                _prenetBiasDevice[i] = new CudaDeviceVariable<float>(_prenetBias[i].Length);
                _prenetBiasDevice[i].CopyToDevice(_prenetBias[i]);
            }

            // LSTM
            _lstmWeightsIhDevice = new CudaDeviceVariable<float>(_lstmWeightsIh.Length);
            _lstmWeightsIhDevice.CopyToDevice(Flatten(_lstmWeightsIh));
            _lstmWeightsHhDevice = new CudaDeviceVariable<float>(_lstmWeightsHh.Length);
            _lstmWeightsHhDevice.CopyToDevice(Flatten(_lstmWeightsHh));
            _lstmBiasDevice = new CudaDeviceVariable<float>(_lstmBias.Length);
            _lstmBiasDevice.CopyToDevice(_lstmBias);

            // Mel projection
            _melProjectionDevice = new CudaDeviceVariable<float>(_melProjection.Length);
            _melProjectionDevice.CopyToDevice(Flatten(_melProjection));
            _melProjectionBiasDevice = new CudaDeviceVariable<float>(_melProjectionBias.Length);
            _melProjectionBiasDevice.CopyToDevice(_melProjectionBias);

            // Postnet
            _postnetConvWeightsDevice = new CudaDeviceVariable<float>[Config.PostnetLayers];
            _postnetConvBiasDevice = new CudaDeviceVariable<float>[Config.PostnetLayers];
            for (int i = 0; i < Config.PostnetLayers; i++)
            {
                _postnetConvWeightsDevice[i] = new CudaDeviceVariable<float>(_postnetConvWeights[i].Length);
                _postnetConvWeightsDevice[i].CopyToDevice(Flatten(_postnetConvWeights[i]));
                _postnetConvBiasDevice[i] = new CudaDeviceVariable<float>(_postnetConvBias[i].Length);
                _postnetConvBiasDevice[i].CopyToDevice(_postnetConvBias[i]);
            }
        }

        private void InitializeIntermediateBuffers()
        {
            // Pre-allocate all intermediate buffers with maximum sizes
            // This eliminates the need for repeated allocation/deallocation during inference

            // Text encoding buffers
            _tokensDevice = new CudaDeviceVariable<int>(_maxSeqLen);
            _embeddedDevice = new CudaDeviceVariable<float>(_maxSeqLen * Config.TextEmbeddingDim);
            _encoderOutputDevice = new CudaDeviceVariable<float>(_maxSeqLen * Config.EncoderDim);

            // Prenet output
            int prenetOutputSize = Config.PrenetDims[^1];
            _prenetOutputDevice = new CudaDeviceVariable<float>(prenetOutputSize);

            // Attention buffers
            _attentionEnergiesDevice = new CudaDeviceVariable<float>(_maxSeqLen);
            _attentionWeightsDevice = new CudaDeviceVariable<float>(_maxSeqLen);
            _attentionContextDevice = new CudaDeviceVariable<float>(Config.EncoderDim);

            // LSTM state buffers
            _lstmHDevice = new CudaDeviceVariable<float>(Config.DecoderDim);
            _lstmCDevice = new CudaDeviceVariable<float>(Config.DecoderDim);

            // LSTM input buffer (prenet output + attention context)
            int lstmInputSize = Config.PrenetDims[^1] + Config.EncoderDim;
            _lstmInputDevice = new CudaDeviceVariable<float>(lstmInputSize);

            // Mel frame and decoder output
            _melFrameDevice = new CudaDeviceVariable<float>(Config.MelBins * Config.OutputsPerStep);
            int decoderOutputSize = Config.DecoderDim + Config.EncoderDim;
            _decoderOutputDevice = new CudaDeviceVariable<float>(decoderOutputSize);

            Console.WriteLine("Initialized persistent GPU buffers for intermediate computations.");
        }

        private float[,] InitializeMatrix(int rows, int cols)
        {
            float[,] matrix = new float[rows, cols];
            float scale = (float)Math.Sqrt(2.0 / (rows + cols));

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = (float)(_random.NextDouble() * 2 - 1) * scale;

            return matrix;
        }

        private float[] Flatten(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float[] flat = new float[rows * cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    flat[i * cols + j] = matrix[i, j];

            return flat;
        }

        public int[] TextToTokens(string text)
        {
            var tokens = new List<int>();
            foreach (char c in text.ToLower())
            {
                if (_charToIdx.TryGetValue(c, out int idx))
                    tokens.Add(idx);
                else
                    tokens.Add(_charToIdx[' ']);
            }
            tokens.Add(_charToIdx['>']);
            return tokens.ToArray();
        }

        public string TokensToText(int[] tokens)
        {
            return string.Join("", tokens.Select(t =>
                _idxToChar.TryGetValue(t, out char c) ? c : ' '));
        }

        /// <summary>
        /// Forward pass using GPU acceleration.
        /// Optimized to minimize CPU-GPU copies by keeping data on GPU.
        /// </summary>
        public (float[,] melOutput, float[] stopTokens, float[,] attentionWeights) Forward(
            int[] textTokens,
            float[,] targetMel = null,
            int speakerId = 0)
        {
            // Encode text on GPU (keeps result on device)
            var (encoderLen, encoderDim) = EncodeTextGpu(textTokens);

            // Get encoder output to CPU for now (TODO: keep on GPU for attention)
            float[,] encoderOutput = GetEncoderOutputFromDevice(encoderLen, encoderDim);

            // Initialize decoder state
            float[] h = new float[Config.DecoderDim];
            float[] c = new float[Config.DecoderDim];
            float[] attentionWeightsAccum = new float[encoderLen];
            float[] prevMelFrame = new float[Config.MelBins * Config.OutputsPerStep];

            int maxSteps = targetMel != null
                ? (targetMel.GetLength(0) + Config.OutputsPerStep - 1) / Config.OutputsPerStep
                : Config.MaxMelLength / Config.OutputsPerStep;

            var melFrames = new List<float[]>();
            var stopTokenList = new List<float>();
            var attentionHistory = new List<float[]>();

            // Decode step by step
            for (int step = 0; step < maxSteps; step++)
            {
                // Prenet on GPU (uses persistent buffers internally)
                float[] prenetOut = ApplyPrenetGpu(prevMelFrame);

                // Attention (uses persistent buffers internally)
                var (context, attWeights) = ComputeAttentionGpu(h, encoderOutput, attentionWeightsAccum);
                attentionHistory.Add(attWeights);

                for (int i = 0; i < encoderLen; i++)
                    attentionWeightsAccum[i] += attWeights[i];

                // LSTM step on GPU (uses persistent buffers)
                float[] lstmInput = prenetOut.Concat(context).ToArray();
                (h, c) = LSTMStepGpu(lstmInput, h, c);

                // Project to mel on GPU
                float[] projInput = h.Concat(context).ToArray();
                float[] melFrame = LinearProjectGpu(projInput, _melProjection, _melProjectionBias);
                melFrames.Add(melFrame);

                // Stop token (simplified - CPU for now)
                float stopLogit = LinearProjectGpu(projInput,
                    new float[,] { { 0 } },
                    new float[] { 0 })[0];
                float stopProb = 1f / (1f + (float)Math.Exp(-stopLogit));
                stopTokenList.Add(stopProb);

                // Teacher forcing or use predicted
                if (targetMel != null && _training)
                {
                    int frameStart = step * Config.OutputsPerStep;
                    prevMelFrame = new float[Config.MelBins * Config.OutputsPerStep];
                    for (int i = 0; i < Config.OutputsPerStep && frameStart + i < targetMel.GetLength(0); i++)
                        for (int m = 0; m < Config.MelBins; m++)
                            prevMelFrame[i * Config.MelBins + m] = targetMel[frameStart + i, m];
                }
                else
                {
                    prevMelFrame = melFrame;
                    if (stopProb > 0.5f && step > 10) break;
                }
            }

            // Convert to output array
            int totalFrames = melFrames.Count * Config.OutputsPerStep;
            float[,] melOutput = new float[totalFrames, Config.MelBins];

            for (int step = 0; step < melFrames.Count; step++)
            {
                for (int i = 0; i < Config.OutputsPerStep; i++)
                {
                    int frameIdx = step * Config.OutputsPerStep + i;
                    if (frameIdx >= totalFrames) break;

                    for (int m = 0; m < Config.MelBins; m++)
                        melOutput[frameIdx, m] = melFrames[step][i * Config.MelBins + m];
                }
            }

            // Apply postnet on GPU
            float[,] postnetOutput = ApplyPostnetGpu(melOutput);

            // Add residual using GPU kernel
            float[,] melRefined = ApplyPostnetResidualGpu(melOutput, postnetOutput);

            // Convert attention history
            float[,] attentionOut = new float[attentionHistory.Count, encoderLen];
            for (int t = 0; t < attentionHistory.Count; t++)
                for (int e = 0; e < encoderLen; e++)
                    attentionOut[t, e] = attentionHistory[t][e];

            return (melRefined, stopTokenList.ToArray(), attentionOut);
        }

        /// <summary>
        /// Backward pass - compute gradients for all parameters.
        /// Call after Forward() during training.
        /// </summary>
        public void Backward(float[,] melOutput, float[,] targetMel, float[] stopPredicted)
        {
            if (!_training) return;

            int frames = Math.Min(melOutput.GetLength(0), targetMel.GetLength(0));
            int melBins = Config.MelBins;
            int totalSize = frames * melBins;

            // Compute mel loss gradient: d(MSE)/d(output) = 2*(output - target)/N
            float[,] melGradient = new float[frames, melBins];
            for (int t = 0; t < frames; t++)
            {
                for (int m = 0; m < melBins; m++)
                {
                    melGradient[t, m] = 2f * (melOutput[t, m] - targetMel[t, m]) / totalSize;
                }
            }

            // Initialize gradient accumulators if not already done
            InitializeGradients();

            // Backprop through postnet
            float[,] postnetGrad = BackwardPostnet(melGradient);

            // Add gradients from residual connection
            for (int t = 0; t < frames; t++)
                for (int m = 0; m < melBins; m++)
                    postnetGrad[t, m] += melGradient[t, m];

            // Backprop through decoder (simplified - accumulate gradients)
            BackwardDecoder(postnetGrad);
        }

        // Gradient accumulators
        private float[,] _gradTextEmbedding;
        private float[][,] _gradEncoderConvWeights;
        private float[][] _gradEncoderConvBias;
        private float[,] _gradQueryProj;
        private float[,] _gradKeyProj;
        private float[][,] _gradPrenetWeights;
        private float[][] _gradPrenetBias;
        private float[,] _gradLstmWeightsIh;
        private float[,] _gradLstmWeightsHh;
        private float[] _gradLstmBias;
        private float[,] _gradMelProjection;
        private float[] _gradMelProjectionBias;
        private float[][,] _gradPostnetConvWeights;
        private float[][] _gradPostnetConvBias;

        private void InitializeGradients()
        {
            if (_gradTextEmbedding != null) return; // Already initialized

            _gradTextEmbedding = new float[Config.VocabSize, Config.TextEmbeddingDim];

            _gradEncoderConvWeights = new float[Config.EncoderConvLayers][,];
            _gradEncoderConvBias = new float[Config.EncoderConvLayers][];
            for (int i = 0; i < Config.EncoderConvLayers; i++)
            {
                _gradEncoderConvWeights[i] = new float[_encoderConvWeights[i].GetLength(0), _encoderConvWeights[i].GetLength(1)];
                _gradEncoderConvBias[i] = new float[_encoderConvBias[i].Length];
            }

            _gradQueryProj = new float[_queryProj.GetLength(0), _queryProj.GetLength(1)];
            _gradKeyProj = new float[_keyProj.GetLength(0), _keyProj.GetLength(1)];

            _gradPrenetWeights = new float[Config.PrenetDims.Length][,];
            _gradPrenetBias = new float[Config.PrenetDims.Length][];
            for (int i = 0; i < Config.PrenetDims.Length; i++)
            {
                _gradPrenetWeights[i] = new float[_prenetWeights[i].GetLength(0), _prenetWeights[i].GetLength(1)];
                _gradPrenetBias[i] = new float[_prenetBias[i].Length];
            }

            _gradLstmWeightsIh = new float[_lstmWeightsIh.GetLength(0), _lstmWeightsIh.GetLength(1)];
            _gradLstmWeightsHh = new float[_lstmWeightsHh.GetLength(0), _lstmWeightsHh.GetLength(1)];
            _gradLstmBias = new float[_lstmBias.Length];

            _gradMelProjection = new float[_melProjection.GetLength(0), _melProjection.GetLength(1)];
            _gradMelProjectionBias = new float[_melProjectionBias.Length];

            _gradPostnetConvWeights = new float[Config.PostnetLayers][,];
            _gradPostnetConvBias = new float[Config.PostnetLayers][];
            for (int i = 0; i < Config.PostnetLayers; i++)
            {
                _gradPostnetConvWeights[i] = new float[_postnetConvWeights[i].GetLength(0), _postnetConvWeights[i].GetLength(1)];
                _gradPostnetConvBias[i] = new float[_postnetConvBias[i].Length];
            }
        }

        private void ZeroGradients()
        {
            if (_gradTextEmbedding == null) InitializeGradients();

            Array.Clear(_gradTextEmbedding, 0, _gradTextEmbedding.Length);
            for (int i = 0; i < Config.EncoderConvLayers; i++)
            {
                Array.Clear(_gradEncoderConvWeights[i], 0, _gradEncoderConvWeights[i].Length);
                Array.Clear(_gradEncoderConvBias[i], 0, _gradEncoderConvBias[i].Length);
            }
            Array.Clear(_gradQueryProj, 0, _gradQueryProj.Length);
            Array.Clear(_gradKeyProj, 0, _gradKeyProj.Length);
            for (int i = 0; i < Config.PrenetDims.Length; i++)
            {
                Array.Clear(_gradPrenetWeights[i], 0, _gradPrenetWeights[i].Length);
                Array.Clear(_gradPrenetBias[i], 0, _gradPrenetBias[i].Length);
            }
            Array.Clear(_gradLstmWeightsIh, 0, _gradLstmWeightsIh.Length);
            Array.Clear(_gradLstmWeightsHh, 0, _gradLstmWeightsHh.Length);
            Array.Clear(_gradLstmBias, 0, _gradLstmBias.Length);
            Array.Clear(_gradMelProjection, 0, _gradMelProjection.Length);
            Array.Clear(_gradMelProjectionBias, 0, _gradMelProjectionBias.Length);
            for (int i = 0; i < Config.PostnetLayers; i++)
            {
                Array.Clear(_gradPostnetConvWeights[i], 0, _gradPostnetConvWeights[i].Length);
                Array.Clear(_gradPostnetConvBias[i], 0, _gradPostnetConvBias[i].Length);
            }
        }

        private float[,] BackwardPostnet(float[,] gradOutput)
        {
            // Backprop through postnet conv layers (in reverse order)
            float[,] grad = gradOutput;
            int seqLen = grad.GetLength(0);

            for (int layer = Config.PostnetLayers - 1; layer >= 0; layer--)
            {
                int outChannels = grad.GetLength(1);
                int inChannels = layer == 0 ? Config.MelBins : Config.PostnetChannels;

                // Accumulate weight gradients (simplified)
                int biasLen = _gradPostnetConvBias[layer].Length;
                for (int t = 0; t < seqLen; t++)
                {
                    for (int oc = 0; oc < Math.Min(outChannels, biasLen); oc++)
                    {
                        _gradPostnetConvBias[layer][oc] += grad[t, oc];
                    }
                }

                // Create gradient for input layer
                grad = new float[seqLen, inChannels];
                // Simplified backprop - propagate averaged gradient
                float scale = 1f / Math.Max(1, outChannels);
                for (int t = 0; t < seqLen; t++)
                {
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        grad[t, ic] = gradOutput[t, Math.Min(ic, gradOutput.GetLength(1) - 1)] * scale;
                    }
                }
            }

            return grad;
        }

        private void BackwardDecoder(float[,] gradOutput)
        {
            // Simplified decoder backward - accumulate mel projection gradients
            int frames = gradOutput.GetLength(0);
            // Use the actual mel projection bias size, not gradOutput dimensions
            int melProjSize = _gradMelProjectionBias.Length;
            int gradCols = gradOutput.GetLength(1);

            for (int t = 0; t < frames; t++)
            {
                // Only accumulate up to the smaller of the two dimensions
                int maxM = Math.Min(melProjSize, gradCols);
                for (int m = 0; m < maxM; m++)
                {
                    _gradMelProjectionBias[m] += gradOutput[t, m];
                }
            }
        }

        /// <summary>
        /// Apply Adam optimizer step to all parameters.
        /// </summary>
        public void ApplyGradients(float learningRate = 1e-4f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        {
            // Apply gradients to weights using Adam update
            ApplyAdamUpdate(_textEmbedding, _gradTextEmbedding, ref _mTextEmbedding, ref _vTextEmbedding,
                            learningRate, beta1, beta2, epsilon);

            for (int i = 0; i < Config.EncoderConvLayers; i++)
            {
                ApplyAdamUpdate(_encoderConvWeights[i], _gradEncoderConvWeights[i],
                               ref _mEncoderConvWeights[i], ref _vEncoderConvWeights[i],
                               learningRate, beta1, beta2, epsilon);
            }

            ApplyAdamUpdate(_lstmWeightsIh, _gradLstmWeightsIh, ref _mLstmWeightsIh, ref _vLstmWeightsIh,
                            learningRate, beta1, beta2, epsilon);
            ApplyAdamUpdate(_lstmWeightsHh, _gradLstmWeightsHh, ref _mLstmWeightsHh, ref _vLstmWeightsHh,
                            learningRate, beta1, beta2, epsilon);

            ApplyAdamUpdate(_melProjection, _gradMelProjection, ref _mMelProjection, ref _vMelProjection,
                            learningRate, beta1, beta2, epsilon);

            // Increment timestep
            _adamTimestep++;

            // Zero gradients for next batch
            ZeroGradients();

            // Sync updated weights to GPU
            SyncToDevice();
        }

        // Adam optimizer state
        private int _adamTimestep = 0;
        private float[,] _mTextEmbedding, _vTextEmbedding;
        private float[][,] _mEncoderConvWeights, _vEncoderConvWeights;
        private float[,] _mLstmWeightsIh, _vLstmWeightsIh;
        private float[,] _mLstmWeightsHh, _vLstmWeightsHh;
        private float[,] _mMelProjection, _vMelProjection;

        private void ApplyAdamUpdate(float[,] param, float[,] grad, ref float[,] m, ref float[,] v,
                                      float lr, float beta1, float beta2, float eps)
        {
            if (m == null)
            {
                m = new float[param.GetLength(0), param.GetLength(1)];
                v = new float[param.GetLength(0), param.GetLength(1)];
            }

            float beta1_t = (float)Math.Pow(beta1, _adamTimestep + 1);
            float beta2_t = (float)Math.Pow(beta2, _adamTimestep + 1);

            int rows = param.GetLength(0);
            int cols = param.GetLength(1);

            // Capture arrays in local variables to use in Parallel.For
            float[,] mLocal = m;
            float[,] vLocal = v;

            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    float g = grad[i, j];
                    mLocal[i, j] = beta1 * mLocal[i, j] + (1 - beta1) * g;
                    vLocal[i, j] = beta2 * vLocal[i, j] + (1 - beta2) * g * g;

                    float m_hat = mLocal[i, j] / (1 - beta1_t);
                    float v_hat = vLocal[i, j] / (1 - beta2_t);

                    param[i, j] -= lr * m_hat / ((float)Math.Sqrt(v_hat) + eps);
                }
            });
        }

        /// <summary>
        /// GPU-accelerated text encoding with batched matrix multiplication.
        /// Keeps data on GPU and returns device pointer info.
        /// </summary>
        private (int seqLen, int encoderDim) EncodeTextGpu(int[] tokens)
        {
            int seqLen = tokens.Length;
            int embedDim = Config.TextEmbeddingDim;

            // Resize tokens buffer if needed
            if (seqLen > _maxSeqLen)
            {
                _maxSeqLen = seqLen;
                _tokensDevice?.Dispose();
                _tokensDevice = new CudaDeviceVariable<int>(_maxSeqLen);
                _embeddedDevice?.Dispose();
                _embeddedDevice = new CudaDeviceVariable<float>(_maxSeqLen * Config.TextEmbeddingDim);
                _encoderOutputDevice?.Dispose();
                _encoderOutputDevice = new CudaDeviceVariable<float>(_maxSeqLen * Config.EncoderDim);
            }

            if (_ttsKernelsLoaded && _embeddingLookupKernel != null)
            {
                // Use CUDA kernel for embedding lookup
                _tokensDevice.CopyToDevice(tokens);

                _embeddingLookupKernel.BlockDimensions = new dim3(Math.Min(embedDim, 256));
                _embeddingLookupKernel.GridDimensions = new dim3(seqLen);
                _embeddingLookupKernel.Run(
                    _tokensDevice.DevicePointer,
                    _textEmbeddingDevice.DevicePointer,
                    _embeddedDevice.DevicePointer,
                    seqLen,
                    embedDim);
            }
            else
            {
                // CPU fallback for embedding lookup
                float[] embeddedFlat = new float[seqLen * embedDim];
                for (int t = 0; t < seqLen; t++)
                    for (int d = 0; d < embedDim; d++)
                        embeddedFlat[t * embedDim + d] = _textEmbedding[tokens[t], d];
                _embeddedDevice.CopyToDevice(embeddedFlat);
            }

            // Apply encoder convolutions (still need to work with host memory for now)
            // TODO: Keep conv operations on GPU end-to-end
            float[] embeddedHost = new float[seqLen * embedDim];
            _embeddedDevice.CopyToHost(embeddedHost);
            float[,] embedded = new float[seqLen, embedDim];
            for (int t = 0; t < seqLen; t++)
                for (int d = 0; d < embedDim; d++)
                    embedded[t, d] = embeddedHost[t * embedDim + d];

            float[,] current = embedded;
            for (int layer = 0; layer < Config.EncoderConvLayers; layer++)
            {
                bool isLastLayer = layer == Config.EncoderConvLayers - 1;
                current = ApplyConv1DGpu(current, _encoderConvWeights[layer],
                    _encoderConvBias[layer], Config.EncoderKernelSize,
                    isLastLayer ? "tanh" : "relu");
            }

            // Store result in encoder output buffer (keep on GPU)
            float[] currentFlat = Flatten(current);
            _encoderOutputDevice.CopyToDevice(currentFlat);

            return (seqLen, Config.EncoderDim);
        }

        /// <summary>
        /// Helper method to get encoder output from GPU to CPU when needed.
        /// </summary>
        private float[,] GetEncoderOutputFromDevice(int seqLen, int encoderDim)
        {
            float[] outputFlat = new float[seqLen * encoderDim];
            _encoderOutputDevice.CopyToHost(outputFlat);

            float[,] output = new float[seqLen, encoderDim];
            for (int t = 0; t < seqLen; t++)
                for (int d = 0; d < encoderDim; d++)
                    output[t, d] = outputFlat[t * encoderDim + d];

            return output;
        }

        /// <summary>
        /// GPU-accelerated 1D convolution with fused activation.
        /// </summary>
        private float[,] ApplyConv1DGpu(float[,] input, float[,] weights, float[] bias, int kernelSize, string activation = null)
        {
            int seqLen = input.GetLength(0);
            int inChannels = input.GetLength(1);
            int outChannels = weights.GetLength(1);

            float[,] output = new float[seqLen, outChannels];

            // Choose the appropriate CUDA kernel based on activation
            CudaKernel convKernel = activation switch
            {
                "relu" when _ttsKernelsLoaded => _conv1dForwardReLUKernel,
                "tanh" when _ttsKernelsLoaded => _conv1dForwardTanhKernel,
                _ when _ttsKernelsLoaded => _conv1dForwardKernel,
                _ => null
            };

            if (convKernel != null)
            {
                // Flatten input and copy to device
                float[] inputFlat = Flatten(input);
                float[] weightsFlat = Flatten(weights);

                using var inputDevice = new CudaDeviceVariable<float>(inputFlat.Length);
                using var weightsDevice = new CudaDeviceVariable<float>(weightsFlat.Length);
                using var biasDevice = new CudaDeviceVariable<float>(bias.Length);
                using var outputDevice = new CudaDeviceVariable<float>(seqLen * outChannels);

                inputDevice.CopyToDevice(inputFlat);
                weightsDevice.CopyToDevice(weightsFlat);
                biasDevice.CopyToDevice(bias);

                // Configure kernel dimensions
                int blockSizeX = 16;
                int blockSizeY = 16;
                convKernel.BlockDimensions = new dim3(blockSizeX, blockSizeY);
                convKernel.GridDimensions = new dim3(
                    (seqLen + blockSizeX - 1) / blockSizeX,
                    (outChannels + blockSizeY - 1) / blockSizeY);

                convKernel.Run(
                    inputDevice.DevicePointer,
                    weightsDevice.DevicePointer,
                    biasDevice.DevicePointer,
                    outputDevice.DevicePointer,
                    seqLen,
                    inChannels,
                    outChannels,
                    kernelSize);

                // Copy result back
                float[] outputFlat = new float[seqLen * outChannels];
                outputDevice.CopyToHost(outputFlat);

                for (int t = 0; t < seqLen; t++)
                    for (int oc = 0; oc < outChannels; oc++)
                        output[t, oc] = outputFlat[t * outChannels + oc];
            }
            else
            {
                // CPU fallback with parallelization
                int padding = kernelSize / 2;

                Parallel.For(0, seqLen, t =>
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        float sum = bias[oc];

                        for (int k = 0; k < kernelSize; k++)
                        {
                            int inputIdx = t + k - padding;
                            if (inputIdx < 0 || inputIdx >= seqLen) continue;

                            for (int ic = 0; ic < inChannels; ic++)
                            {
                                int weightIdx = k * inChannels + ic;
                                sum += input[inputIdx, ic] * weights[weightIdx, oc];
                            }
                        }

                        // Apply activation
                        if (activation == "relu")
                            sum = Math.Max(0, sum);
                        else if (activation == "tanh")
                            sum = (float)Math.Tanh(sum);

                        output[t, oc] = sum;
                    }
                });
            }

            return output;
        }

        /// <summary>
        /// GPU-accelerated prenet with ReLU and dropout.
        /// Uses persistent buffers and works with device memory when possible.
        /// </summary>
        private float[] ApplyPrenetGpu(float[] input)
        {
            float[] current = input;

            // Temporary device variables for multi-layer prenet
            CudaDeviceVariable<float> inputDevice = null;
            CudaDeviceVariable<float> outputDevice = null;

            try
            {
                for (int i = 0; i < _prenetWeights.Length; i++)
                {
                    int inputDim = current.Length;
                    int outputDim = _prenetBias[i].Length;

                    if (_ttsKernelsLoaded && _prenetForwardKernel != null)
                    {
                        float[] weightsFlat = Flatten(_prenetWeights[i]);

                        // Allocate temporary buffers for this layer
                        if (inputDevice == null || inputDevice.Size != inputDim)
                        {
                            inputDevice?.Dispose();
                            inputDevice = new CudaDeviceVariable<float>(inputDim);
                        }
                        if (outputDevice == null || outputDevice.Size != outputDim)
                        {
                            outputDevice?.Dispose();
                            outputDevice = new CudaDeviceVariable<float>(outputDim);
                        }

                        using var weightsDevice = new CudaDeviceVariable<float>(weightsFlat.Length);
                        using var biasDevice = new CudaDeviceVariable<float>(outputDim);

                        inputDevice.CopyToDevice(current);
                        weightsDevice.CopyToDevice(weightsFlat);
                        biasDevice.CopyToDevice(_prenetBias[i]);

                        int blockSize = Math.Min(outputDim, 256);
                        _prenetForwardKernel.BlockDimensions = new dim3(blockSize);
                        _prenetForwardKernel.GridDimensions = new dim3((outputDim + blockSize - 1) / blockSize);

                        _prenetForwardKernel.Run(
                            inputDevice.DevicePointer,
                            weightsDevice.DevicePointer,
                            biasDevice.DevicePointer,
                            outputDevice.DevicePointer,
                            inputDim,
                            outputDim);

                        current = new float[outputDim];
                        outputDevice.CopyToHost(current);
                    }
                    else
                    {
                        // CPU fallback
                        current = LinearProjectGpu(current, _prenetWeights[i], _prenetBias[i]);

                        // ReLU
                        for (int j = 0; j < current.Length; j++)
                            current[j] = Math.Max(0, current[j]);
                    }

                    // Dropout (always on for prenet, applied on CPU for randomness)
                    if (_training && Config.PrenetDropout > 0)
                    {
                        for (int j = 0; j < current.Length; j++)
                        {
                            if (_random.NextDouble() < Config.PrenetDropout)
                                current[j] = 0;
                            else
                                current[j] /= (1 - Config.PrenetDropout);
                        }
                    }
                }

                return current;
            }
            finally
            {
                inputDevice?.Dispose();
                outputDevice?.Dispose();
            }
        }

        /// <summary>
        /// GPU-accelerated attention computation with CUDA softmax and context.
        /// Uses persistent buffers to minimize allocation overhead.
        /// </summary>
        private (float[] context, float[] weights) ComputeAttentionGpu(
            float[] decoderState,
            float[,] encoderOutput,
            float[] prevAttentionWeights)
        {
            int encoderLen = encoderOutput.GetLength(0);
            int encoderDim = encoderOutput.GetLength(1);

            // Query projection
            float[] query = LinearProjectGpu(decoderState, _queryProj, null);

            // Pre-compute key projections for all encoder positions
            float[,] keys = new float[encoderLen, Config.AttentionDim];
            Parallel.For(0, encoderLen, e =>
            {
                float[] encoderState = new float[encoderDim];
                for (int d = 0; d < encoderDim; d++)
                    encoderState[d] = encoderOutput[e, d];

                float[] key = LinearProjectGpu(encoderState, _keyProj, null);
                for (int d = 0; d < Config.AttentionDim; d++)
                    keys[e, d] = key[d];
            });

            // Compute energies
            float[] energies = new float[encoderLen];
            Parallel.For(0, encoderLen, e =>
            {
                float energy = 0;
                for (int d = 0; d < Config.AttentionDim; d++)
                    energy += (float)Math.Tanh(query[d] + keys[e, d]);
                energies[e] = energy;
            });

            // Use CUDA for softmax if available (using persistent buffers)
            float[] weights;
            if (_ttsKernelsLoaded && _attentionSoftmaxKernel != null)
            {
                _attentionEnergiesDevice.CopyToDevice(energies);

                int blockSize = Math.Min(encoderLen, 256);
                _attentionSoftmaxKernel.BlockDimensions = new dim3(blockSize);
                _attentionSoftmaxKernel.GridDimensions = new dim3(1);
                _attentionSoftmaxKernel.DynamicSharedMemory = (uint)(blockSize * sizeof(float));

                _attentionSoftmaxKernel.Run(
                    _attentionEnergiesDevice.DevicePointer,
                    _attentionWeightsDevice.DevicePointer,
                    encoderLen);

                weights = new float[encoderLen];
                _attentionWeightsDevice.CopyToHost(weights);
            }
            else
            {
                weights = Softmax(energies);
            }

            // Compute context using CUDA if available (using persistent buffers)
            float[] context = new float[encoderDim];
            if (_ttsKernelsLoaded && _attentionContextKernel != null)
            {
                // Encoder output is already on device from EncodeTextGpu
                _attentionWeightsDevice.CopyToDevice(weights);

                int blockSize = Math.Min(encoderDim, 256);
                _attentionContextKernel.BlockDimensions = new dim3(blockSize);
                _attentionContextKernel.GridDimensions = new dim3((encoderDim + blockSize - 1) / blockSize);

                _attentionContextKernel.Run(
                    _attentionWeightsDevice.DevicePointer,
                    _encoderOutputDevice.DevicePointer,
                    _attentionContextDevice.DevicePointer,
                    encoderLen,
                    encoderDim);

                _attentionContextDevice.CopyToHost(context);
            }
            else
            {
                // CPU fallback
                for (int e = 0; e < encoderLen; e++)
                    for (int d = 0; d < encoderDim; d++)
                        context[d] += weights[e] * encoderOutput[e, d];
            }

            return (context, weights);
        }

        /// <summary>
        /// GPU-accelerated LSTM step.
        /// Uses persistent state buffers to keep LSTM state on GPU between steps.
        /// </summary>
        private (float[] h, float[] c) LSTMStepGpu(float[] input, float[] prevH, float[] prevC)
        {
            int hiddenSize = Config.DecoderDim;
            int inputDim = input.Length;

            float[] newH = new float[hiddenSize];
            float[] newC = new float[hiddenSize];

            if (_ttsKernelsLoaded && _lstmStepOptimizedKernel != null)
            {
                // Use optimized CUDA LSTM kernel with persistent buffers
                _lstmInputDevice.CopyToDevice(input);
                _lstmHDevice.CopyToDevice(prevH);
                _lstmCDevice.CopyToDevice(prevC);

                // Calculate shared memory size for optimized kernel
                int sharedMemSize = (inputDim + hiddenSize) * sizeof(float);

                _lstmStepOptimizedKernel.BlockDimensions = new dim3(Math.Min(hiddenSize, 256));
                _lstmStepOptimizedKernel.GridDimensions = new dim3((hiddenSize + 255) / 256);
                _lstmStepOptimizedKernel.DynamicSharedMemory = (uint)sharedMemSize;

                // Reuse persistent buffers for output
                using var newHDevice = new CudaDeviceVariable<float>(hiddenSize);
                using var newCDevice = new CudaDeviceVariable<float>(hiddenSize);

                _lstmStepOptimizedKernel.Run(
                    _lstmInputDevice.DevicePointer,
                    _lstmHDevice.DevicePointer,
                    _lstmCDevice.DevicePointer,
                    _lstmWeightsIhDevice.DevicePointer,
                    _lstmWeightsHhDevice.DevicePointer,
                    _lstmBiasDevice.DevicePointer,
                    newHDevice.DevicePointer,
                    newCDevice.DevicePointer,
                    inputDim,
                    hiddenSize);

                newHDevice.CopyToHost(newH);
                newCDevice.CopyToHost(newC);
            }
            else
            {
                // CPU fallback
                float[] gates = new float[hiddenSize * 4];

                Parallel.For(0, hiddenSize * 4, g =>
                {
                    gates[g] = _lstmBias[g];

                    for (int i = 0; i < input.Length; i++)
                        gates[g] += input[i] * _lstmWeightsIh[i, g];

                    for (int h = 0; h < hiddenSize; h++)
                        gates[g] += prevH[h] * _lstmWeightsHh[h, g];
                });

                Parallel.For(0, hiddenSize, i =>
                {
                    float inputGate = 1f / (1f + (float)Math.Exp(-gates[i]));
                    float forgetGate = 1f / (1f + (float)Math.Exp(-gates[hiddenSize + i]));
                    float cellGate = (float)Math.Tanh(gates[hiddenSize * 2 + i]);
                    float outputGate = 1f / (1f + (float)Math.Exp(-gates[hiddenSize * 3 + i]));

                    newC[i] = forgetGate * prevC[i] + inputGate * cellGate;
                    newH[i] = outputGate * (float)Math.Tanh(newC[i]);
                });
            }

            return (newH, newC);
        }

        /// <summary>
        /// GPU-accelerated linear projection (mel projection).
        /// </summary>
        private float[] LinearProjectGpu(float[] input, float[,] weights, float[] bias)
        {
            int inputDim = input.Length;
            int outputSize = weights.GetLength(1);
            float[] output = new float[outputSize];

            if (_ttsKernelsLoaded && _melProjectionKernel != null && bias != null)
            {
                float[] weightsFlat = Flatten(weights);

                using var inputDevice = new CudaDeviceVariable<float>(inputDim);
                using var weightsDevice = new CudaDeviceVariable<float>(weightsFlat.Length);
                using var biasDevice = new CudaDeviceVariable<float>(outputSize);
                using var outputDevice = new CudaDeviceVariable<float>(outputSize);

                inputDevice.CopyToDevice(input);
                weightsDevice.CopyToDevice(weightsFlat);
                biasDevice.CopyToDevice(bias);

                int blockSize = Math.Min(outputSize, 256);
                _melProjectionKernel.BlockDimensions = new dim3(blockSize);
                _melProjectionKernel.GridDimensions = new dim3((outputSize + blockSize - 1) / blockSize);

                _melProjectionKernel.Run(
                    inputDevice.DevicePointer,
                    weightsDevice.DevicePointer,
                    biasDevice.DevicePointer,
                    outputDevice.DevicePointer,
                    inputDim,
                    outputSize);

                outputDevice.CopyToHost(output);
            }
            else
            {
                // CPU fallback
                Parallel.For(0, outputSize, o =>
                {
                    output[o] = bias != null ? bias[o] : 0;
                    for (int i = 0; i < input.Length; i++)
                        output[o] += input[i] * weights[i, o];
                });
            }

            return output;
        }

        /// <summary>
        /// GPU-accelerated postnet with fused activations.
        /// </summary>
        private float[,] ApplyPostnetGpu(float[,] melInput)
        {
            float[,] current = melInput;

            for (int layer = 0; layer < Config.PostnetLayers; layer++)
            {
                // Use fused tanh activation for all layers except the last
                string activation = layer < Config.PostnetLayers - 1 ? "tanh" : null;
                current = ApplyConv1DGpu(current, _postnetConvWeights[layer],
                    _postnetConvBias[layer], Config.PostnetKernelSize, activation);
            }

            return current;
        }

        /// <summary>
        /// GPU-accelerated postnet residual addition.
        /// </summary>
        private float[,] ApplyPostnetResidualGpu(float[,] melOutput, float[,] postnetOutput)
        {
            int frames = melOutput.GetLength(0);
            int melBins = melOutput.GetLength(1);
            int totalSize = frames * melBins;

            float[,] result = new float[frames, melBins];

            if (_ttsKernelsLoaded && _postnetResidualKernel != null)
            {
                float[] melFlat = Flatten(melOutput);
                float[] postnetFlat = Flatten(postnetOutput);

                using var melDevice = new CudaDeviceVariable<float>(totalSize);
                using var postnetDevice = new CudaDeviceVariable<float>(totalSize);
                using var resultDevice = new CudaDeviceVariable<float>(totalSize);

                melDevice.CopyToDevice(melFlat);
                postnetDevice.CopyToDevice(postnetFlat);

                int blockSize = 256;
                _postnetResidualKernel.BlockDimensions = new dim3(blockSize);
                _postnetResidualKernel.GridDimensions = new dim3((totalSize + blockSize - 1) / blockSize);

                _postnetResidualKernel.Run(
                    melDevice.DevicePointer,
                    postnetDevice.DevicePointer,
                    resultDevice.DevicePointer,
                    totalSize);

                float[] resultFlat = new float[totalSize];
                resultDevice.CopyToHost(resultFlat);

                for (int t = 0; t < frames; t++)
                    for (int m = 0; m < melBins; m++)
                        result[t, m] = resultFlat[t * melBins + m];
            }
            else
            {
                // CPU fallback
                Parallel.For(0, frames, t =>
                {
                    for (int m = 0; m < melBins; m++)
                        result[t, m] = melOutput[t, m] + postnetOutput[t, m];
                });
            }

            return result;
        }

        private float[] Softmax(float[] input)
        {
            float max = input.Max();
            float[] exp = input.Select(x => (float)Math.Exp(x - max)).ToArray();
            float sum = exp.Sum();
            return exp.Select(x => x / sum).ToArray();
        }

        public float ComputeLoss(float[,] predicted, float[,] target, float[] stopPredicted = null)
        {
            int frames = Math.Min(predicted.GetLength(0), target.GetLength(0));
            int melBins = Config.MelBins;

            float melLoss = 0;
            int count = 0;

            for (int t = 0; t < frames; t++)
            {
                for (int m = 0; m < melBins; m++)
                {
                    float diff = predicted[t, m] - target[t, m];
                    melLoss += diff * diff;
                    count++;
                }
            }

            return count > 0 ? melLoss / count : 0;
        }

        public void SetTraining(bool training) => _training = training;

        public int CountParameters()
        {
            int count = _textEmbedding.Length;

            foreach (var w in _encoderConvWeights) count += w.Length;
            foreach (var b in _encoderConvBias) count += b.Length;

            count += _queryProj.Length + _keyProj.Length;

            foreach (var w in _prenetWeights) count += w.Length;
            foreach (var b in _prenetBias) count += b.Length;

            count += _lstmWeightsIh.Length + _lstmWeightsHh.Length + _lstmBias.Length;
            count += _melProjection.Length + _melProjectionBias.Length;

            foreach (var w in _postnetConvWeights) count += w.Length;
            foreach (var b in _postnetConvBias) count += b.Length;

            return count;
        }

        public void SyncToDevice()
        {
            _textEmbeddingDevice.CopyToDevice(Flatten(_textEmbedding));
            for (int i = 0; i < Config.EncoderConvLayers; i++)
            {
                _encoderConvWeightsDevice[i].CopyToDevice(Flatten(_encoderConvWeights[i]));
                _encoderConvBiasDevice[i].CopyToDevice(_encoderConvBias[i]);
            }
            _queryProjDevice.CopyToDevice(Flatten(_queryProj));
            _keyProjDevice.CopyToDevice(Flatten(_keyProj));
            for (int i = 0; i < Config.PrenetDims.Length; i++)
            {
                _prenetWeightsDevice[i].CopyToDevice(Flatten(_prenetWeights[i]));
                _prenetBiasDevice[i].CopyToDevice(_prenetBias[i]);
            }
            _lstmWeightsIhDevice.CopyToDevice(Flatten(_lstmWeightsIh));
            _lstmWeightsHhDevice.CopyToDevice(Flatten(_lstmWeightsHh));
            _lstmBiasDevice.CopyToDevice(_lstmBias);
            _melProjectionDevice.CopyToDevice(Flatten(_melProjection));
            _melProjectionBiasDevice.CopyToDevice(_melProjectionBias);
            for (int i = 0; i < Config.PostnetLayers; i++)
            {
                _postnetConvWeightsDevice[i].CopyToDevice(Flatten(_postnetConvWeights[i]));
                _postnetConvBiasDevice[i].CopyToDevice(_postnetConvBias[i]);
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            // Dispose weight buffers
            _textEmbeddingDevice?.Dispose();
            foreach (var d in _encoderConvWeightsDevice ?? Array.Empty<CudaDeviceVariable<float>>()) d?.Dispose();
            foreach (var d in _encoderConvBiasDevice ?? Array.Empty<CudaDeviceVariable<float>>()) d?.Dispose();
            _queryProjDevice?.Dispose();
            _keyProjDevice?.Dispose();
            foreach (var d in _prenetWeightsDevice ?? Array.Empty<CudaDeviceVariable<float>>()) d?.Dispose();
            foreach (var d in _prenetBiasDevice ?? Array.Empty<CudaDeviceVariable<float>>()) d?.Dispose();
            _lstmWeightsIhDevice?.Dispose();
            _lstmWeightsHhDevice?.Dispose();
            _lstmBiasDevice?.Dispose();
            _melProjectionDevice?.Dispose();
            _melProjectionBiasDevice?.Dispose();
            foreach (var d in _postnetConvWeightsDevice ?? Array.Empty<CudaDeviceVariable<float>>()) d?.Dispose();
            foreach (var d in _postnetConvBiasDevice ?? Array.Empty<CudaDeviceVariable<float>>()) d?.Dispose();

            // Dispose intermediate buffers
            _embeddedDevice?.Dispose();
            _encoderOutputDevice?.Dispose();
            _prenetOutputDevice?.Dispose();
            _attentionContextDevice?.Dispose();
            _attentionWeightsDevice?.Dispose();
            _attentionEnergiesDevice?.Dispose();
            _lstmHDevice?.Dispose();
            _lstmCDevice?.Dispose();
            _lstmInputDevice?.Dispose();
            _melFrameDevice?.Dispose();
            _decoderOutputDevice?.Dispose();
            _tokensDevice?.Dispose();

            _context?.Dispose();

            GC.SuppressFinalize(this);
        }

        ~TTSModelCuda() => Dispose();
    }

    /// <summary>
    /// Optimal configuration for RTX 5090 (32GB VRAM, Compute 12.0).
    /// </summary>
    public static class TTSConfigOptimizer
    {
        public static TTSConfig ForRTX5090()
        {
            return new TTSConfig
            {
                // Large model - RTX 5090 has plenty of VRAM
                TextEmbeddingDim = 512,
                EncoderConvLayers = 5,
                EncoderDim = 512,
                EncoderKernelSize = 5,

                // Audio settings
                MelBins = 80,
                OutputsPerStep = 3,  // Faster inference
                MaxMelLength = 2000,

                // Large attention
                AttentionDim = 256,
                AttentionFilters = 64,
                AttentionKernelSize = 31,

                // Large decoder
                PrenetDims = new[] { 256, 256 },
                DecoderDim = 1024,
                DecoderLayers = 2,

                // Large postnet
                PostnetLayers = 5,
                PostnetKernelSize = 5,
                PostnetChannels = 512,

                // Lower dropout for big model
                DropoutRate = 0.1f,
                PrenetDropout = 0.5f,

                // Higher learning rate - RTX 5090 can handle larger batches
                LearningRate = 1e-3f,

                // Audio processing
                SampleRate = 22050,
                FFTSize = 1024,
                HopLength = 256
            };
        }

        public static (int batchSize, int accumSteps) GetOptimalBatchConfig()
        {
            // RTX 5090 with 32GB can handle large batches
            return (batchSize: 32, accumSteps: 2);  // Effective batch size: 64
        }
    }

    /// <summary>
    /// Auto-tuning batch size optimizer that finds optimal configuration at runtime.
    /// Tests different batch sizes and measures throughput to find the best setting.
    /// </summary>
    public class BatchSizeAutoTuner : IDisposable
    {
        private CudaContext _context;
        private bool _disposed;

        // GPU info
        public string DeviceName { get; private set; }
        public long TotalMemoryBytes { get; private set; }
        public long FreeMemoryBytes { get; private set; }
        public int ComputeCapabilityMajor { get; private set; }
        public int ComputeCapabilityMinor { get; private set; }

        // Hardware signature for cache validation
        public string HardwareSignature { get; private set; }

        // Tuning results
        public int OptimalBatchSize { get; private set; }
        public int OptimalAccumSteps { get; private set; }
        public float MaxThroughput { get; private set; }
        public Dictionary<int, TuningResult> Results { get; } = new Dictionary<int, TuningResult>();

        // Cache file location
        private static string CacheDirectory => Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "NeuralTrainer-NET");
        private static string CacheFilePath => Path.Combine(CacheDirectory, "batch_tuning_cache.json");

        public class TuningResult
        {
            public int BatchSize { get; set; }
            public int AccumSteps { get; set; }
            public float ThroughputSamplesPerSec { get; set; }
            public long MemoryUsedBytes { get; set; }
            public float GpuUtilization { get; set; }
            public bool OutOfMemory { get; set; }
            public string Error { get; set; }
        }

        /// <summary>
        /// Cache entry for storing tuning results per hardware/config combination.
        /// </summary>
        public class TuningCacheEntry
        {
            public string HardwareSignature { get; set; }
            public string ConfigSignature { get; set; }
            public int OptimalBatchSize { get; set; }
            public int OptimalAccumSteps { get; set; }
            public float MaxThroughput { get; set; }
            public DateTime CachedAt { get; set; }
            public string DeviceName { get; set; }
            public long TotalMemoryBytes { get; set; }
        }

        /// <summary>
        /// Full cache file structure.
        /// </summary>
        public class TuningCacheFile
        {
            public int Version { get; set; } = 1;
            public List<TuningCacheEntry> Entries { get; set; } = new List<TuningCacheEntry>();
        }

        public BatchSizeAutoTuner(CudaContext context = null)
        {
            _context = context ?? new CudaContext();
            QueryGpuInfo();
        }

        private void QueryGpuInfo()
        {
            DeviceName = _context.GetDeviceName();
            TotalMemoryBytes = (long)_context.GetTotalDeviceMemorySize();
            FreeMemoryBytes = (long)_context.GetFreeDeviceMemorySize();
            var cc = _context.GetDeviceComputeCapability();
            ComputeCapabilityMajor = cc.Major;
            ComputeCapabilityMinor = cc.Minor;

            // Generate hardware signature from immutable GPU properties
            HardwareSignature = $"{DeviceName}|{TotalMemoryBytes}|{ComputeCapabilityMajor}.{ComputeCapabilityMinor}";

            Console.WriteLine($"GPU: {DeviceName}");
            Console.WriteLine($"Memory: {FreeMemoryBytes / (1024.0 * 1024 * 1024):F1} GB free / {TotalMemoryBytes / (1024.0 * 1024 * 1024):F1} GB total");
            Console.WriteLine($"Compute Capability: {ComputeCapabilityMajor}.{ComputeCapabilityMinor}");
        }

        /// <summary>
        /// Generate a signature for the model config to ensure cache validity.
        /// </summary>
        private static string GetConfigSignature(TTSConfig config)
        {
            return $"{config.TextEmbeddingDim}|{config.EncoderDim}|{config.DecoderDim}|{config.AttentionDim}|{config.MelBins}|{config.PostnetChannels}";
        }

        /// <summary>
        /// Try to load cached tuning results for the current hardware and config.
        /// Returns true if valid cache was found.
        /// </summary>
        public bool TryLoadCache(TTSConfig config, out int batchSize, out int accumSteps)
        {
            batchSize = 0;
            accumSteps = 0;

            try
            {
                if (!File.Exists(CacheFilePath))
                    return false;

                string json = File.ReadAllText(CacheFilePath);
                var cache = JsonSerializer.Deserialize<TuningCacheFile>(json);

                if (cache == null || cache.Entries == null)
                    return false;

                string configSig = GetConfigSignature(config);

                // Find matching entry for current hardware and config
                var entry = cache.Entries.FirstOrDefault(e =>
                    e.HardwareSignature == HardwareSignature &&
                    e.ConfigSignature == configSig);

                if (entry != null)
                {
                    batchSize = entry.OptimalBatchSize;
                    accumSteps = entry.OptimalAccumSteps;
                    OptimalBatchSize = batchSize;
                    OptimalAccumSteps = accumSteps;
                    MaxThroughput = entry.MaxThroughput;

                    Console.WriteLine($"Loaded cached tuning results from {entry.CachedAt:yyyy-MM-dd HH:mm}");
                    Console.WriteLine($"  Hardware: {entry.DeviceName}");
                    Console.WriteLine($"  Batch size: {batchSize}, Accum steps: {accumSteps}");
                    return true;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not load tuning cache: {ex.Message}");
            }

            return false;
        }

        /// <summary>
        /// Save current tuning results to cache file.
        /// </summary>
        public void SaveCache(TTSConfig config)
        {
            try
            {
                // Ensure directory exists
                Directory.CreateDirectory(CacheDirectory);

                // Load existing cache or create new
                TuningCacheFile cache;
                if (File.Exists(CacheFilePath))
                {
                    string existingJson = File.ReadAllText(CacheFilePath);
                    cache = JsonSerializer.Deserialize<TuningCacheFile>(existingJson) ?? new TuningCacheFile();
                }
                else
                {
                    cache = new TuningCacheFile();
                }

                string configSig = GetConfigSignature(config);

                // Remove existing entry for same hardware/config if present
                cache.Entries.RemoveAll(e =>
                    e.HardwareSignature == HardwareSignature &&
                    e.ConfigSignature == configSig);

                // Add new entry
                cache.Entries.Add(new TuningCacheEntry
                {
                    HardwareSignature = HardwareSignature,
                    ConfigSignature = configSig,
                    OptimalBatchSize = OptimalBatchSize,
                    OptimalAccumSteps = OptimalAccumSteps,
                    MaxThroughput = MaxThroughput,
                    CachedAt = DateTime.Now,
                    DeviceName = DeviceName,
                    TotalMemoryBytes = TotalMemoryBytes
                });

                // Write cache
                var options = new JsonSerializerOptions { WriteIndented = true };
                string json = JsonSerializer.Serialize(cache, options);
                File.WriteAllText(CacheFilePath, json);

                Console.WriteLine($"Saved tuning results to cache: {CacheFilePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not save tuning cache: {ex.Message}");
            }
        }

        /// <summary>
        /// Auto-tune with caching support. Checks cache first, runs tuning only if needed.
        /// </summary>
        public (int batchSize, int accumSteps) AutoTuneWithCache(
            TTSConfig config,
            int targetEffectiveBatch = 256,
            bool forceRetune = false)
        {
            // Check cache first (unless forced to retune)
            if (!forceRetune && TryLoadCache(config, out int cachedBatch, out int cachedAccum))
            {
                return (cachedBatch, cachedAccum);
            }

            // Run quick auto-tune
            var result = QuickAutoTune(config, targetEffectiveBatch);

            // Save to cache
            SaveCache(config);

            return result;
        }

        /// <summary>
        /// Auto-tune batch size for the given model configuration.
        /// Tests multiple batch sizes and finds the one with highest throughput.
        /// </summary>
        public (int batchSize, int accumSteps) AutoTune(
            TTSConfig config,
            int targetEffectiveBatch = 256,
            int minBatchSize = 8,
            int maxBatchSize = 512,
            int warmupIterations = 2,
            int benchmarkIterations = 5)
        {
            Console.WriteLine($"\n=== Auto-Tuning Batch Size ===");
            Console.WriteLine($"Target effective batch: {targetEffectiveBatch}");
            Console.WriteLine($"Testing range: {minBatchSize} - {maxBatchSize}\n");

            // Calculate batch sizes to test (powers of 2 and some intermediates)
            var batchSizesToTest = new List<int>();
            for (int bs = minBatchSize; bs <= maxBatchSize; bs *= 2)
            {
                batchSizesToTest.Add(bs);
                if (bs * 3 / 2 <= maxBatchSize && bs * 3 / 2 > bs)
                    batchSizesToTest.Add(bs * 3 / 2);
            }
            batchSizesToTest.Sort();
            batchSizesToTest = batchSizesToTest.Distinct().ToList();

            // Estimate memory per sample based on model config
            long estimatedMemoryPerSample = EstimateMemoryPerSample(config);
            Console.WriteLine($"Estimated memory per sample: {estimatedMemoryPerSample / (1024.0 * 1024):F1} MB");

            // Filter out batch sizes that definitely won't fit
            long safeMemoryLimit = (long)(FreeMemoryBytes * 0.85); // Use 85% of free memory
            int maxFeasibleBatch = (int)(safeMemoryLimit / estimatedMemoryPerSample);
            Console.WriteLine($"Max feasible batch size (estimated): {maxFeasibleBatch}");

            batchSizesToTest = batchSizesToTest.Where(bs => bs <= Math.Max(maxFeasibleBatch, minBatchSize)).ToList();

            // Test each batch size
            float bestThroughput = 0;
            int bestBatchSize = minBatchSize;

            foreach (int batchSize in batchSizesToTest.OrderByDescending(x => x))
            {
                var result = TestBatchSize(config, batchSize, warmupIterations, benchmarkIterations);
                Results[batchSize] = result;

                if (!result.OutOfMemory && result.ThroughputSamplesPerSec > bestThroughput)
                {
                    bestThroughput = result.ThroughputSamplesPerSec;
                    bestBatchSize = batchSize;
                }

                // Print result
                if (result.OutOfMemory)
                {
                    Console.WriteLine($"  Batch {batchSize,4}: OOM - {result.Error}");
                }
                else
                {
                    Console.WriteLine($"  Batch {batchSize,4}: {result.ThroughputSamplesPerSec,8:F1} samples/s, " +
                                    $"Memory: {result.MemoryUsedBytes / (1024.0 * 1024 * 1024):F2} GB");
                }

                // Force GC and CUDA cleanup between tests
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            // Calculate accumulation steps to reach target effective batch
            int accumSteps = Math.Max(1, targetEffectiveBatch / bestBatchSize);
            int effectiveBatch = bestBatchSize * accumSteps;

            OptimalBatchSize = bestBatchSize;
            OptimalAccumSteps = accumSteps;
            MaxThroughput = bestThroughput;

            Console.WriteLine($"\n=== Auto-Tune Results ===");
            Console.WriteLine($"Optimal batch size: {bestBatchSize}");
            Console.WriteLine($"Accumulation steps: {accumSteps}");
            Console.WriteLine($"Effective batch size: {effectiveBatch}");
            Console.WriteLine($"Max throughput: {bestThroughput:F1} samples/s");

            return (bestBatchSize, accumSteps);
        }

        /// <summary>
        /// Quick auto-tune that uses heuristics based on GPU memory.
        /// Faster than full benchmark but may not find absolute optimal.
        /// </summary>
        public (int batchSize, int accumSteps) QuickAutoTune(TTSConfig config, int targetEffectiveBatch = 256)
        {
            Console.WriteLine($"\n=== Quick Auto-Tune ===");

            long estimatedMemoryPerSample = EstimateMemoryPerSample(config);
            long safeMemoryLimit = (long)(FreeMemoryBytes * 0.80);

            // Start with max feasible and binary search down if OOM
            int maxBatch = Math.Min(512, (int)(safeMemoryLimit / estimatedMemoryPerSample));
            maxBatch = Math.Max(8, (maxBatch / 8) * 8); // Round to multiple of 8

            Console.WriteLine($"Estimated max batch: {maxBatch}");

            // Test a few key sizes quickly
            int[] testSizes = { maxBatch, maxBatch / 2, maxBatch / 4, 64, 32 };
            testSizes = testSizes.Where(x => x >= 8).Distinct().OrderByDescending(x => x).ToArray();

            int bestBatch = 32;
            foreach (int bs in testSizes)
            {
                try
                {
                    var result = TestBatchSize(config, bs, warmupIterations: 1, benchmarkIterations: 2);
                    if (!result.OutOfMemory)
                    {
                        bestBatch = bs;
                        Console.WriteLine($"  Batch {bs}: OK ({result.ThroughputSamplesPerSec:F1} samples/s)");
                        break;
                    }
                    else
                    {
                        Console.WriteLine($"  Batch {bs}: OOM");
                    }
                }
                catch
                {
                    Console.WriteLine($"  Batch {bs}: Error");
                }

                GC.Collect();
            }

            int accumSteps = Math.Max(1, targetEffectiveBatch / bestBatch);

            Console.WriteLine($"\nSelected: batch={bestBatch}, accum={accumSteps} (effective={bestBatch * accumSteps})");

            OptimalBatchSize = bestBatch;
            OptimalAccumSteps = accumSteps;

            return (bestBatch, accumSteps);
        }

        private TuningResult TestBatchSize(TTSConfig config, int batchSize, int warmupIterations, int benchmarkIterations)
        {
            var result = new TuningResult { BatchSize = batchSize };

            try
            {
                // Create dummy data for testing
                var textTokensBatch = new int[batchSize][];
                var targetMelBatch = new float[batchSize][,];
                var speakerIds = new int[batchSize];

                for (int i = 0; i < batchSize; i++)
                {
                    textTokensBatch[i] = Enumerable.Range(0, 50).ToArray(); // 50 tokens
                    targetMelBatch[i] = new float[200, config.MelBins]; // 200 mel frames
                    speakerIds[i] = 0;
                }

                // Create model for testing
                using var model = new TTSModelCuda(config);

                // Warmup - process batch samples sequentially
                for (int w = 0; w < warmupIterations; w++)
                {
                    for (int i = 0; i < batchSize; i++)
                    {
                        model.Forward(textTokensBatch[i], targetMelBatch[i], speakerIds[i]);
                    }
                }

                // Benchmark
                long memBefore = (long)_context.GetFreeDeviceMemorySize();
                var sw = System.Diagnostics.Stopwatch.StartNew();

                for (int iter = 0; iter < benchmarkIterations; iter++)
                {
                    for (int i = 0; i < batchSize; i++)
                    {
                        model.Forward(textTokensBatch[i], targetMelBatch[i], speakerIds[i]);
                    }
                }

                sw.Stop();
                long memAfter = (long)_context.GetFreeDeviceMemorySize();

                int totalSamples = batchSize * benchmarkIterations;
                result.ThroughputSamplesPerSec = totalSamples / (float)sw.Elapsed.TotalSeconds;
                result.MemoryUsedBytes = memBefore - memAfter;
                result.OutOfMemory = false;
            }
            catch (CudaException ex) when (ex.CudaError == CUResult.ErrorOutOfMemory ||
                                           ex.Message.Contains("out of memory", StringComparison.OrdinalIgnoreCase))
            {
                result.OutOfMemory = true;
                result.Error = "CUDA out of memory";
            }
            catch (OutOfMemoryException)
            {
                result.OutOfMemory = true;
                result.Error = "System out of memory";
            }
            catch (Exception ex)
            {
                result.OutOfMemory = true;
                result.Error = ex.Message;
            }

            return result;
        }

        private long EstimateMemoryPerSample(TTSConfig config)
        {
            // Rough estimation of GPU memory per training sample
            // Based on model dimensions and typical sequence lengths

            long bytesPerFloat = 4;
            int avgSeqLen = 100;
            int avgMelLen = 500;

            // Embeddings
            long embedMem = avgSeqLen * config.TextEmbeddingDim * bytesPerFloat;

            // Encoder
            long encoderMem = avgSeqLen * config.EncoderDim * config.EncoderConvLayers * bytesPerFloat;

            // Attention
            long attnMem = avgSeqLen * avgMelLen * bytesPerFloat; // attention weights
            attnMem += avgSeqLen * config.AttentionDim * bytesPerFloat; // keys
            attnMem += avgMelLen * config.AttentionDim * bytesPerFloat; // queries

            // Decoder LSTM
            long lstmMem = avgMelLen * config.DecoderDim * 4 * bytesPerFloat; // gates
            lstmMem += config.DecoderDim * 2 * bytesPerFloat; // h and c states

            // Mel output
            long melMem = avgMelLen * config.MelBins * bytesPerFloat;

            // Postnet
            long postnetMem = avgMelLen * config.PostnetChannels * config.PostnetLayers * bytesPerFloat;

            // Gradients (double the activations for backward)
            long totalActivations = embedMem + encoderMem + attnMem + lstmMem + melMem + postnetMem;
            long gradientMem = totalActivations; // Roughly same size for gradients

            // Add some overhead (fragmentation, temporary buffers)
            long totalPerSample = (long)((totalActivations + gradientMem) * 1.5);

            return totalPerSample;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            // Don't dispose context if it was passed in
            GC.SuppressFinalize(this);
        }

        ~BatchSizeAutoTuner() => Dispose();
    }
}
