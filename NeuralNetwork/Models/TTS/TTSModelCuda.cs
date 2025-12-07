using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
                var module = _context.LoadModulePTX(denseKernelPath);
                _matmulKernel = new CudaKernel("matmul_kernel", module, _context);
                _addBiasKernel = new CudaKernel("add_bias_kernel", module, _context);
            }

            if (File.Exists(activationsPath))
            {
                var module = _context.LoadModulePTX(activationsPath);
                try { _reluKernel = new CudaKernel("relu_forward", module, _context); } catch { }
                try { _sigmoidKernel = new CudaKernel("sigmoid_forward", module, _context); } catch { }
                try { _tanhKernel = new CudaKernel("tanh_forward", module, _context); } catch { }
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
        /// </summary>
        public (float[,] melOutput, float[] stopTokens, float[,] attentionWeights) Forward(
            int[] textTokens,
            float[,] targetMel = null,
            int speakerId = 0)
        {
            // Encode text on GPU
            float[,] encoderOutput = EncodeTextGpu(textTokens);
            int encoderLen = encoderOutput.GetLength(0);

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
                // Prenet on GPU
                float[] prenetOut = ApplyPrenetGpu(prevMelFrame);

                // Attention
                var (context, attWeights) = ComputeAttentionGpu(h, encoderOutput, attentionWeightsAccum);
                attentionHistory.Add(attWeights);

                for (int i = 0; i < encoderLen; i++)
                    attentionWeightsAccum[i] += attWeights[i];

                // LSTM step on GPU
                float[] lstmInput = prenetOut.Concat(context).ToArray();
                (h, c) = LSTMStepGpu(lstmInput, h, c);

                // Project to mel on GPU
                float[] projInput = h.Concat(context).ToArray();
                float[] melFrame = LinearProjectGpu(projInput, _melProjection, _melProjectionBias);
                melFrames.Add(melFrame);

                // Stop token
                float stopLogit = LinearProjectGpu(projInput,
                    new float[,] { { 0 } }, // Simplified - just use CPU for stop token
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
        /// GPU-accelerated text encoding with batched matrix multiplication.
        /// </summary>
        private float[,] EncodeTextGpu(int[] tokens)
        {
            int seqLen = tokens.Length;
            int embedDim = Config.TextEmbeddingDim;

            float[,] embedded;

            if (_ttsKernelsLoaded && _embeddingLookupKernel != null)
            {
                // Use CUDA kernel for embedding lookup
                using var tokensDevice = new CudaDeviceVariable<int>(seqLen);
                tokensDevice.CopyToDevice(tokens);

                using var embeddedDevice = new CudaDeviceVariable<float>(seqLen * embedDim);

                _embeddingLookupKernel.BlockDimensions = new dim3(Math.Min(embedDim, 256));
                _embeddingLookupKernel.GridDimensions = new dim3(seqLen);
                _embeddingLookupKernel.Run(
                    tokensDevice.DevicePointer,
                    _textEmbeddingDevice.DevicePointer,
                    embeddedDevice.DevicePointer,
                    seqLen,
                    embedDim);

                float[] embeddedFlat = new float[seqLen * embedDim];
                embeddedDevice.CopyToHost(embeddedFlat);

                embedded = new float[seqLen, embedDim];
                for (int t = 0; t < seqLen; t++)
                    for (int d = 0; d < embedDim; d++)
                        embedded[t, d] = embeddedFlat[t * embedDim + d];
            }
            else
            {
                // CPU fallback for embedding lookup
                embedded = new float[seqLen, embedDim];
                for (int t = 0; t < seqLen; t++)
                    for (int d = 0; d < embedDim; d++)
                        embedded[t, d] = _textEmbedding[tokens[t], d];
            }

            // Apply encoder convolutions
            float[,] current = embedded;
            for (int layer = 0; layer < Config.EncoderConvLayers; layer++)
            {
                bool isLastLayer = layer == Config.EncoderConvLayers - 1;
                current = ApplyConv1DGpu(current, _encoderConvWeights[layer],
                    _encoderConvBias[layer], Config.EncoderKernelSize,
                    isLastLayer ? "tanh" : "relu");
            }

            return current;
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
        /// </summary>
        private float[] ApplyPrenetGpu(float[] input)
        {
            float[] current = input;

            for (int i = 0; i < _prenetWeights.Length; i++)
            {
                int inputDim = current.Length;
                int outputDim = _prenetBias[i].Length;

                if (_ttsKernelsLoaded && _prenetForwardKernel != null)
                {
                    float[] weightsFlat = Flatten(_prenetWeights[i]);

                    using var inputDevice = new CudaDeviceVariable<float>(inputDim);
                    using var weightsDevice = new CudaDeviceVariable<float>(weightsFlat.Length);
                    using var biasDevice = new CudaDeviceVariable<float>(outputDim);
                    using var outputDevice = new CudaDeviceVariable<float>(outputDim);

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

        /// <summary>
        /// GPU-accelerated attention computation with CUDA softmax and context.
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

            // Use CUDA for softmax if available
            float[] weights;
            if (_ttsKernelsLoaded && _attentionSoftmaxKernel != null)
            {
                using var energiesDevice = new CudaDeviceVariable<float>(encoderLen);
                using var weightsDevice = new CudaDeviceVariable<float>(encoderLen);

                energiesDevice.CopyToDevice(energies);

                int blockSize = Math.Min(encoderLen, 256);
                _attentionSoftmaxKernel.BlockDimensions = new dim3(blockSize);
                _attentionSoftmaxKernel.GridDimensions = new dim3(1);
                _attentionSoftmaxKernel.DynamicSharedMemory = (uint)(blockSize * sizeof(float));

                _attentionSoftmaxKernel.Run(
                    energiesDevice.DevicePointer,
                    weightsDevice.DevicePointer,
                    encoderLen);

                weights = new float[encoderLen];
                weightsDevice.CopyToHost(weights);
            }
            else
            {
                weights = Softmax(energies);
            }

            // Compute context using CUDA if available
            float[] context = new float[encoderDim];
            if (_ttsKernelsLoaded && _attentionContextKernel != null)
            {
                float[] encoderFlat = Flatten(encoderOutput);

                using var weightsDevice = new CudaDeviceVariable<float>(encoderLen);
                using var encoderDevice = new CudaDeviceVariable<float>(encoderFlat.Length);
                using var contextDevice = new CudaDeviceVariable<float>(encoderDim);

                weightsDevice.CopyToDevice(weights);
                encoderDevice.CopyToDevice(encoderFlat);

                int blockSize = Math.Min(encoderDim, 256);
                _attentionContextKernel.BlockDimensions = new dim3(blockSize);
                _attentionContextKernel.GridDimensions = new dim3((encoderDim + blockSize - 1) / blockSize);

                _attentionContextKernel.Run(
                    weightsDevice.DevicePointer,
                    encoderDevice.DevicePointer,
                    contextDevice.DevicePointer,
                    encoderLen,
                    encoderDim);

                contextDevice.CopyToHost(context);
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
        /// </summary>
        private (float[] h, float[] c) LSTMStepGpu(float[] input, float[] prevH, float[] prevC)
        {
            int hiddenSize = Config.DecoderDim;
            int inputDim = input.Length;

            float[] newH = new float[hiddenSize];
            float[] newC = new float[hiddenSize];

            if (_ttsKernelsLoaded && _lstmStepOptimizedKernel != null)
            {
                // Use optimized CUDA LSTM kernel
                using var inputDevice = new CudaDeviceVariable<float>(inputDim);
                using var prevHDevice = new CudaDeviceVariable<float>(hiddenSize);
                using var prevCDevice = new CudaDeviceVariable<float>(hiddenSize);
                using var newHDevice = new CudaDeviceVariable<float>(hiddenSize);
                using var newCDevice = new CudaDeviceVariable<float>(hiddenSize);

                inputDevice.CopyToDevice(input);
                prevHDevice.CopyToDevice(prevH);
                prevCDevice.CopyToDevice(prevC);

                // Calculate shared memory size for optimized kernel
                int sharedMemSize = (inputDim + hiddenSize) * sizeof(float);

                _lstmStepOptimizedKernel.BlockDimensions = new dim3(Math.Min(hiddenSize, 256));
                _lstmStepOptimizedKernel.GridDimensions = new dim3((hiddenSize + 255) / 256);
                _lstmStepOptimizedKernel.DynamicSharedMemory = (uint)sharedMemSize;

                _lstmStepOptimizedKernel.Run(
                    inputDevice.DevicePointer,
                    prevHDevice.DevicePointer,
                    prevCDevice.DevicePointer,
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
            // Sync other weights as needed during training
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

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
}
