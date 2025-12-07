using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Models.TTS
{
    /// <summary>
    /// Tacotron-style Text-to-Speech model.
    /// Architecture: Text Encoder → Attention → Mel Decoder → Postnet
    /// </summary>
    public class TTSModel
    {
        public TTSConfig Config { get; }

        // Character vocabulary
        private Dictionary<char, int> _charToIdx;
        private Dictionary<int, char> _idxToChar;

        // Encoder weights
        private float[,] _textEmbedding;           // [vocab_size, embed_dim]
        private float[][,] _encoderConvWeights;    // Conv1D weights [layers][kernel, in, out]
        private float[][] _encoderConvBias;

        // Attention weights
        private float[,] _queryProj;               // [decoder_dim, attention_dim]
        private float[,] _keyProj;                 // [encoder_dim, attention_dim]
        private float[,] _locationConv;            // Location-sensitive conv
        private float[] _locationBias;
        private float[,] _attentionV;              // [attention_dim, 1]

        // Decoder weights
        private float[][,] _prenetWeights;         // Prenet layers
        private float[][] _prenetBias;
        private float[,] _lstmWeightsIh;           // LSTM input weights
        private float[,] _lstmWeightsHh;           // LSTM hidden weights
        private float[] _lstmBiasIh;
        private float[] _lstmBiasHh;
        private float[,] _melProjection;           // Project to mel bins
        private float[] _melProjectionBias;
        private float[,] _stopProjection;          // Stop token prediction
        private float[] _stopProjectionBias;

        // Postnet weights
        private float[][,] _postnetConvWeights;
        private float[][] _postnetConvBias;

        // Speaker embedding (optional)
        private float[,] _speakerEmbedding;

        // Training state
        private bool _training = true;
        private Random _random;

        // Gradient storage
        private Dictionary<string, float[,]> _gradients2D = new();
        private Dictionary<string, float[]> _gradients1D = new();

        public TTSModel(TTSConfig config, int? seed = null)
        {
            Config = config;
            Config.Validate();

            _random = seed.HasValue ? new Random(seed.Value) : new Random();

            InitializeVocabulary();
            InitializeWeights();
        }

        private void InitializeVocabulary()
        {
            // Create character vocabulary (ASCII printable + special chars)
            _charToIdx = new Dictionary<char, int>();
            _idxToChar = new Dictionary<int, char>();

            int idx = 0;
            // Special tokens
            _charToIdx['<'] = idx; _idxToChar[idx++] = '<';  // PAD
            _charToIdx['>'] = idx; _idxToChar[idx++] = '>';  // EOS

            // Standard ASCII printable characters
            for (char c = ' '; c <= '~'; c++)
            {
                if (!_charToIdx.ContainsKey(c))
                {
                    _charToIdx[c] = idx;
                    _idxToChar[idx++] = c;
                }
            }

            // Update vocab size
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
            _locationConv = InitializeMatrix(Config.AttentionKernelSize, Config.AttentionFilters);
            _locationBias = new float[Config.AttentionFilters];
            _attentionV = InitializeMatrix(Config.AttentionDim + Config.AttentionFilters, 1);

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

            // Decoder LSTM (simplified single layer)
            int lstmInputDim = Config.PrenetDims[^1] + Config.EncoderDim;
            if (Config.NumSpeakers > 0) lstmInputDim += Config.SpeakerEmbeddingDim;

            _lstmWeightsIh = InitializeMatrix(lstmInputDim, Config.DecoderDim * 4);
            _lstmWeightsHh = InitializeMatrix(Config.DecoderDim, Config.DecoderDim * 4);
            _lstmBiasIh = new float[Config.DecoderDim * 4];
            _lstmBiasHh = new float[Config.DecoderDim * 4];

            // Initialize forget gate bias to 1 for better gradient flow
            for (int i = Config.DecoderDim; i < Config.DecoderDim * 2; i++)
            {
                _lstmBiasIh[i] = 1f;
            }

            // Mel projection
            int projInput = Config.DecoderDim + Config.EncoderDim;
            _melProjection = InitializeMatrix(projInput, Config.MelBins * Config.OutputsPerStep);
            _melProjectionBias = new float[Config.MelBins * Config.OutputsPerStep];

            // Stop projection
            _stopProjection = InitializeMatrix(projInput, 1);
            _stopProjectionBias = new float[1];

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

            // Speaker embedding
            if (Config.NumSpeakers > 0)
            {
                _speakerEmbedding = InitializeMatrix(Config.NumSpeakers, Config.SpeakerEmbeddingDim);
            }
        }

        private float[,] InitializeMatrix(int rows, int cols)
        {
            float[,] matrix = new float[rows, cols];
            float scale = (float)Math.Sqrt(2.0 / (rows + cols));

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = (float)(_random.NextDouble() * 2 - 1) * scale;
                }
            }

            return matrix;
        }

        /// <summary>
        /// Convert text to token indices.
        /// </summary>
        public int[] TextToTokens(string text)
        {
            var tokens = new List<int>();
            foreach (char c in text.ToLower())
            {
                if (_charToIdx.TryGetValue(c, out int idx))
                    tokens.Add(idx);
                else
                    tokens.Add(_charToIdx[' ']); // Unknown -> space
            }
            tokens.Add(_charToIdx['>']); // EOS
            return tokens.ToArray();
        }

        /// <summary>
        /// Convert tokens back to text.
        /// </summary>
        public string TokensToText(int[] tokens)
        {
            return string.Join("", tokens.Select(t =>
                _idxToChar.TryGetValue(t, out char c) ? c : ' '));
        }

        /// <summary>
        /// Forward pass: text -> mel spectrogram.
        /// </summary>
        /// <param name="textTokens">Input text tokens [seq_len]</param>
        /// <param name="targetMel">Target mel spectrogram for teacher forcing [time, mel_bins] (null for inference)</param>
        /// <param name="speakerId">Speaker ID for multi-speaker model</param>
        /// <returns>Predicted mel spectrogram, stop tokens, attention weights</returns>
        public (float[,] melOutput, float[] stopTokens, float[,] attentionWeights) Forward(
            int[] textTokens,
            float[,] targetMel = null,
            int speakerId = 0)
        {
            // 1. Encode text
            float[,] encoderOutput = EncodeText(textTokens);
            int encoderLen = encoderOutput.GetLength(0);

            // 2. Get speaker embedding if multi-speaker
            float[] speakerEmbed = null;
            if (Config.NumSpeakers > 0 && _speakerEmbedding != null)
            {
                speakerEmbed = new float[Config.SpeakerEmbeddingDim];
                for (int i = 0; i < Config.SpeakerEmbeddingDim; i++)
                {
                    speakerEmbed[i] = _speakerEmbedding[speakerId, i];
                }
            }

            // 3. Initialize decoder state
            float[] h = new float[Config.DecoderDim];  // LSTM hidden
            float[] c = new float[Config.DecoderDim];  // LSTM cell
            float[] attentionWeightsAccum = new float[encoderLen];  // Accumulated attention
            float[] prevMelFrame = new float[Config.MelBins * Config.OutputsPerStep];

            // 4. Determine output length
            int maxSteps = targetMel != null
                ? (targetMel.GetLength(0) + Config.OutputsPerStep - 1) / Config.OutputsPerStep
                : Config.MaxMelLength / Config.OutputsPerStep;

            var melFrames = new List<float[]>();
            var stopTokenList = new List<float>();
            var attentionHistory = new List<float[]>();

            // 5. Decode step by step
            for (int step = 0; step < maxSteps; step++)
            {
                // Prenet
                float[] prenetOut = ApplyPrenet(prevMelFrame);

                // Compute attention context
                var (context, attWeights) = ComputeAttention(
                    h, encoderOutput, attentionWeightsAccum);
                attentionHistory.Add(attWeights);

                // Update accumulated attention weights
                for (int i = 0; i < encoderLen; i++)
                    attentionWeightsAccum[i] += attWeights[i];

                // Prepare LSTM input: concat(prenet, context, speaker_embed)
                var lstmInput = new List<float>(prenetOut);
                lstmInput.AddRange(context);
                if (speakerEmbed != null)
                    lstmInput.AddRange(speakerEmbed);

                // LSTM step
                (h, c) = LSTMStep(lstmInput.ToArray(), h, c);

                // Project to mel output
                float[] projInput = h.Concat(context).ToArray();
                float[] melFrame = LinearProject(projInput, _melProjection, _melProjectionBias);
                melFrames.Add(melFrame);

                // Stop token prediction
                float stopLogit = LinearProject(projInput, _stopProjection, _stopProjectionBias)[0];
                float stopProb = Sigmoid(stopLogit);
                stopTokenList.Add(stopProb);

                // Teacher forcing or use predicted frame
                if (targetMel != null && _training)
                {
                    int frameStart = step * Config.OutputsPerStep;
                    prevMelFrame = new float[Config.MelBins * Config.OutputsPerStep];
                    for (int i = 0; i < Config.OutputsPerStep && frameStart + i < targetMel.GetLength(0); i++)
                    {
                        for (int m = 0; m < Config.MelBins; m++)
                        {
                            prevMelFrame[i * Config.MelBins + m] = targetMel[frameStart + i, m];
                        }
                    }
                }
                else
                {
                    prevMelFrame = melFrame;

                    // Early stop during inference
                    if (stopProb > 0.5f && step > 10)
                        break;
                }
            }

            // 6. Convert to mel output array
            int totalFrames = melFrames.Count * Config.OutputsPerStep;
            float[,] melOutput = new float[totalFrames, Config.MelBins];

            for (int step = 0; step < melFrames.Count; step++)
            {
                for (int i = 0; i < Config.OutputsPerStep; i++)
                {
                    int frameIdx = step * Config.OutputsPerStep + i;
                    if (frameIdx >= totalFrames) break;

                    for (int m = 0; m < Config.MelBins; m++)
                    {
                        melOutput[frameIdx, m] = melFrames[step][i * Config.MelBins + m];
                    }
                }
            }

            // 7. Apply postnet refinement
            float[,] melRefined = ApplyPostnet(melOutput);

            // 8. Add residual
            for (int t = 0; t < totalFrames; t++)
            {
                for (int m = 0; m < Config.MelBins; m++)
                {
                    melRefined[t, m] += melOutput[t, m];
                }
            }

            // Convert attention history to 2D array
            float[,] attentionOut = new float[attentionHistory.Count, encoderLen];
            for (int t = 0; t < attentionHistory.Count; t++)
            {
                for (int e = 0; e < encoderLen; e++)
                {
                    attentionOut[t, e] = attentionHistory[t][e];
                }
            }

            return (melRefined, stopTokenList.ToArray(), attentionOut);
        }

        /// <summary>
        /// Encode text using character embedding and convolutions.
        /// </summary>
        private float[,] EncodeText(int[] tokens)
        {
            int seqLen = tokens.Length;

            // Embed characters
            float[,] embedded = new float[seqLen, Config.TextEmbeddingDim];
            for (int t = 0; t < seqLen; t++)
            {
                for (int d = 0; d < Config.TextEmbeddingDim; d++)
                {
                    embedded[t, d] = _textEmbedding[tokens[t], d];
                }
            }

            // Apply encoder convolutions
            float[,] current = embedded;
            for (int layer = 0; layer < Config.EncoderConvLayers; layer++)
            {
                current = ApplyConv1D(current, _encoderConvWeights[layer],
                    _encoderConvBias[layer], Config.EncoderKernelSize);

                // ReLU activation (except last layer uses tanh)
                int rows = current.GetLength(0);
                int cols = current.GetLength(1);
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        current[i, j] = layer < Config.EncoderConvLayers - 1
                            ? Math.Max(0, current[i, j])  // ReLU
                            : (float)Math.Tanh(current[i, j]);  // Tanh
                    }
                }

                // Dropout
                if (_training && Config.DropoutRate > 0)
                {
                    current = ApplyDropout(current, Config.DropoutRate);
                }
            }

            return current;
        }

        /// <summary>
        /// Apply 1D convolution.
        /// </summary>
        private float[,] ApplyConv1D(float[,] input, float[,] weights, float[] bias, int kernelSize)
        {
            int seqLen = input.GetLength(0);
            int inChannels = input.GetLength(1);
            int outChannels = weights.GetLength(1);
            int padding = kernelSize / 2;

            float[,] output = new float[seqLen, outChannels];

            for (int t = 0; t < seqLen; t++)
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

                    output[t, oc] = sum;
                }
            }

            return output;
        }

        /// <summary>
        /// Apply prenet (MLP with dropout).
        /// </summary>
        private float[] ApplyPrenet(float[] input)
        {
            float[] current = input;

            for (int i = 0; i < _prenetWeights.Length; i++)
            {
                current = LinearProject(current, _prenetWeights[i], _prenetBias[i]);

                // ReLU
                for (int j = 0; j < current.Length; j++)
                    current[j] = Math.Max(0, current[j]);

                // Dropout (always on, even during inference - important for prenet)
                if (Config.PrenetDropout > 0)
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
        /// Compute attention context and weights.
        /// </summary>
        private (float[] context, float[] weights) ComputeAttention(
            float[] decoderState,
            float[,] encoderOutput,
            float[] prevAttentionWeights)
        {
            int encoderLen = encoderOutput.GetLength(0);
            int encoderDim = encoderOutput.GetLength(1);

            // Query projection
            float[] query = LinearProject(decoderState, _queryProj, null);

            // Key projection for all encoder positions
            float[] energies = new float[encoderLen];

            for (int e = 0; e < encoderLen; e++)
            {
                // Get encoder state
                float[] encoderState = new float[encoderDim];
                for (int d = 0; d < encoderDim; d++)
                    encoderState[d] = encoderOutput[e, d];

                // Key projection
                float[] key = LinearProject(encoderState, _keyProj, null);

                // Location features (simplified)
                float[] locationFeatures = new float[Config.AttentionFilters];
                for (int f = 0; f < Config.AttentionFilters; f++)
                {
                    float sum = _locationBias[f];
                    for (int k = 0; k < Config.AttentionKernelSize; k++)
                    {
                        int idx = e + k - Config.AttentionKernelSize / 2;
                        if (idx >= 0 && idx < encoderLen)
                        {
                            sum += prevAttentionWeights[idx] * _locationConv[k, f];
                        }
                    }
                    locationFeatures[f] = sum;
                }

                // Combine and compute energy
                float[] combined = new float[Config.AttentionDim + Config.AttentionFilters];
                for (int d = 0; d < Config.AttentionDim; d++)
                {
                    combined[d] = (float)Math.Tanh(query[d] + key[d]);
                }
                for (int f = 0; f < Config.AttentionFilters; f++)
                {
                    combined[Config.AttentionDim + f] = locationFeatures[f];
                }

                // Final energy score
                energies[e] = 0;
                for (int d = 0; d < combined.Length; d++)
                {
                    energies[e] += combined[d] * _attentionV[d, 0];
                }
            }

            // Softmax
            float[] weights = Softmax(energies);

            // Compute context
            float[] context = new float[encoderDim];
            for (int e = 0; e < encoderLen; e++)
            {
                for (int d = 0; d < encoderDim; d++)
                {
                    context[d] += weights[e] * encoderOutput[e, d];
                }
            }

            return (context, weights);
        }

        /// <summary>
        /// Single LSTM step.
        /// </summary>
        private (float[] h, float[] c) LSTMStep(float[] input, float[] prevH, float[] prevC)
        {
            int hiddenSize = Config.DecoderDim;

            // Input transformation
            float[] gates = new float[hiddenSize * 4];

            // Input to gates
            for (int g = 0; g < hiddenSize * 4; g++)
            {
                gates[g] = _lstmBiasIh[g] + _lstmBiasHh[g];

                for (int i = 0; i < input.Length; i++)
                {
                    gates[g] += input[i] * _lstmWeightsIh[i, g];
                }

                for (int h = 0; h < hiddenSize; h++)
                {
                    gates[g] += prevH[h] * _lstmWeightsHh[h, g];
                }
            }

            // Split gates
            float[] newC = new float[hiddenSize];
            float[] newH = new float[hiddenSize];

            for (int i = 0; i < hiddenSize; i++)
            {
                float inputGate = Sigmoid(gates[i]);
                float forgetGate = Sigmoid(gates[hiddenSize + i]);
                float cellGate = (float)Math.Tanh(gates[hiddenSize * 2 + i]);
                float outputGate = Sigmoid(gates[hiddenSize * 3 + i]);

                newC[i] = forgetGate * prevC[i] + inputGate * cellGate;
                newH[i] = outputGate * (float)Math.Tanh(newC[i]);
            }

            return (newH, newC);
        }

        /// <summary>
        /// Apply postnet convolutions.
        /// </summary>
        private float[,] ApplyPostnet(float[,] melInput)
        {
            float[,] current = melInput;

            for (int layer = 0; layer < Config.PostnetLayers; layer++)
            {
                current = ApplyConv1D(current, _postnetConvWeights[layer],
                    _postnetConvBias[layer], Config.PostnetKernelSize);

                // Activation: tanh for all except last layer (linear)
                if (layer < Config.PostnetLayers - 1)
                {
                    int rows = current.GetLength(0);
                    int cols = current.GetLength(1);
                    for (int i = 0; i < rows; i++)
                    {
                        for (int j = 0; j < cols; j++)
                        {
                            current[i, j] = (float)Math.Tanh(current[i, j]);
                        }
                    }

                    if (_training && Config.DropoutRate > 0)
                    {
                        current = ApplyDropout(current, Config.DropoutRate);
                    }
                }
            }

            return current;
        }

        private float[] LinearProject(float[] input, float[,] weights, float[] bias)
        {
            int outputSize = weights.GetLength(1);
            float[] output = new float[outputSize];

            for (int o = 0; o < outputSize; o++)
            {
                output[o] = bias != null ? bias[o] : 0;
                for (int i = 0; i < input.Length; i++)
                {
                    output[o] += input[i] * weights[i, o];
                }
            }

            return output;
        }

        private float[,] ApplyDropout(float[,] input, float rate)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);
            float[,] output = new float[rows, cols];
            float scale = 1f / (1f - rate);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (_random.NextDouble() >= rate)
                        output[i, j] = input[i, j] * scale;
                }
            }

            return output;
        }

        private float Sigmoid(float x)
        {
            return 1f / (1f + (float)Math.Exp(-x));
        }

        private float[] Softmax(float[] input)
        {
            float max = input.Max();
            float[] exp = input.Select(x => (float)Math.Exp(x - max)).ToArray();
            float sum = exp.Sum();
            return exp.Select(x => x / sum).ToArray();
        }

        /// <summary>
        /// Compute loss between predicted and target mel spectrograms.
        /// </summary>
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

            melLoss = count > 0 ? melLoss / count : 0;

            // Stop token loss (binary cross-entropy)
            float stopLoss = 0;
            if (stopPredicted != null)
            {
                int targetFrames = target.GetLength(0);
                for (int t = 0; t < stopPredicted.Length; t++)
                {
                    int frameEnd = (t + 1) * Config.OutputsPerStep;
                    float targetStop = frameEnd >= targetFrames ? 1f : 0f;
                    float pred = Math.Clamp(stopPredicted[t], 1e-7f, 1f - 1e-7f);
                    stopLoss -= targetStop * (float)Math.Log(pred) +
                               (1 - targetStop) * (float)Math.Log(1 - pred);
                }
                stopLoss /= stopPredicted.Length;
            }

            return melLoss + 0.1f * stopLoss;
        }

        /// <summary>
        /// Set training mode.
        /// </summary>
        public void SetTraining(bool training)
        {
            _training = training;
        }

        /// <summary>
        /// Get all parameters for optimizer.
        /// </summary>
        public IEnumerable<(string name, float[,] param)> GetParameters2D()
        {
            yield return ("textEmbedding", _textEmbedding);

            for (int i = 0; i < _encoderConvWeights.Length; i++)
                yield return ($"encoderConv{i}", _encoderConvWeights[i]);

            yield return ("queryProj", _queryProj);
            yield return ("keyProj", _keyProj);
            yield return ("locationConv", _locationConv);
            yield return ("attentionV", _attentionV);

            for (int i = 0; i < _prenetWeights.Length; i++)
                yield return ($"prenet{i}", _prenetWeights[i]);

            yield return ("lstmWeightsIh", _lstmWeightsIh);
            yield return ("lstmWeightsHh", _lstmWeightsHh);
            yield return ("melProjection", _melProjection);
            yield return ("stopProjection", _stopProjection);

            for (int i = 0; i < _postnetConvWeights.Length; i++)
                yield return ($"postnet{i}", _postnetConvWeights[i]);

            if (_speakerEmbedding != null)
                yield return ("speakerEmbedding", _speakerEmbedding);
        }

        /// <summary>
        /// Count total parameters.
        /// </summary>
        public int CountParameters()
        {
            int count = 0;

            count += _textEmbedding.Length;

            foreach (var w in _encoderConvWeights) count += w.Length;
            foreach (var b in _encoderConvBias) count += b.Length;

            count += _queryProj.Length;
            count += _keyProj.Length;
            count += _locationConv.Length;
            count += _locationBias.Length;
            count += _attentionV.Length;

            foreach (var w in _prenetWeights) count += w.Length;
            foreach (var b in _prenetBias) count += b.Length;

            count += _lstmWeightsIh.Length;
            count += _lstmWeightsHh.Length;
            count += _lstmBiasIh.Length;
            count += _lstmBiasHh.Length;

            count += _melProjection.Length;
            count += _melProjectionBias.Length;
            count += _stopProjection.Length;
            count += _stopProjectionBias.Length;

            foreach (var w in _postnetConvWeights) count += w.Length;
            foreach (var b in _postnetConvBias) count += b.Length;

            if (_speakerEmbedding != null) count += _speakerEmbedding.Length;

            return count;
        }
    }
}
