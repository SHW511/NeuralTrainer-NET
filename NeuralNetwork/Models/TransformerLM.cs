using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NeuralNetwork.Tensors;
using NeuralNetwork.Layers;
using NeuralNetwork.Layers.Attention;
using NeuralNetwork.Initializers;

namespace NeuralNetwork.Models
{
    /// <summary>
    /// Transformer Language Model (decoder-only, like GPT).
    ///
    /// Architecture:
    /// 1. Token Embedding
    /// 2. Positional Encoding
    /// 3. N x Transformer Blocks
    /// 4. Final Layer Normalization
    /// 5. Output Projection (optional weight tying with embeddings)
    /// </summary>
    public class TransformerLM
    {
        private readonly TransformerConfig _config;

        // Model components
        private float[,] _tokenEmbeddings;      // [vocabSize, embeddingDim]
        private PositionalEncoding _posEncoding;
        private Dropout _embeddingDropout;
        private TransformerBlock[] _blocks;
        private LayerNorm _finalNorm;
        private float[,] _outputProjection;     // [embeddingDim, vocabSize] (or tied to embeddings)

        // Gradient accumulators
        private float[,] _tokenEmbeddingsGrad;
        private float[,] _outputProjectionGrad;

        // Cached values for backward pass
        private int[] _lastTokens;
        private Tensor _lastEmbedded;
        private Tensor _lastAfterPosEnc;
        private Tensor[] _blockOutputs;
        private Tensor _lastNormOutput;

        private bool _training;

        public TransformerConfig Config => _config;

        public bool Training
        {
            get => _training;
            set
            {
                _training = value;
                _posEncoding.Training = value;
                if (_embeddingDropout != null) _embeddingDropout.Training = value;
                foreach (var block in _blocks)
                    block.Training = value;
            }
        }

        // Expose parameters for optimizer
        public float[,] TokenEmbeddings => _tokenEmbeddings;
        public float[,] TokenEmbeddingsGrad => _tokenEmbeddingsGrad;
        public float[,] OutputProjection => _outputProjection;
        public float[,] OutputProjectionGrad => _outputProjectionGrad;
        public TransformerBlock[] Blocks => _blocks;
        public LayerNorm FinalNorm => _finalNorm;

        /// <summary>
        /// Create a Transformer Language Model.
        /// </summary>
        public TransformerLM(TransformerConfig config)
        {
            _config = config;
            config.Validate();
            _training = true;

            Initialize();
        }

        private void Initialize()
        {
            // Token embeddings
            _tokenEmbeddings = InitializeEmbeddings(_config.VocabSize, _config.EmbeddingDim);
            _tokenEmbeddingsGrad = new float[_config.VocabSize, _config.EmbeddingDim];

            // Positional encoding
            _posEncoding = new PositionalEncoding(_config.MaxSeqLen, _config.EmbeddingDim, _config.EmbeddingDropout);

            // Embedding dropout (applied after positional encoding)
            if (_config.EmbeddingDropout > 0f)
            {
                _embeddingDropout = new Dropout(_config.EmbeddingDropout);
            }

            // Transformer blocks
            _blocks = new TransformerBlock[_config.NumLayers];
            for (int i = 0; i < _config.NumLayers; i++)
            {
                _blocks[i] = new TransformerBlock(
                    _config.EmbeddingDim,
                    _config.NumHeads,
                    _config.FFNDim,
                    _config.DropoutRate,
                    _config.UseCausalMask
                );
                // Build the block so weights are initialized
                _blocks[i].Build(new[] { 1, _config.MaxSeqLen, _config.EmbeddingDim });
            }

            // Final layer normalization
            _finalNorm = new LayerNorm(_config.EmbeddingDim);

            // Output projection
            if (_config.TieEmbeddings)
            {
                // Use transposed embeddings as output projection
                _outputProjection = null;
                _outputProjectionGrad = null;
            }
            else
            {
                _outputProjection = Initializers.Initializers.GlorotUniform(_config.EmbeddingDim, _config.VocabSize);
                _outputProjectionGrad = new float[_config.EmbeddingDim, _config.VocabSize];
            }

            _blockOutputs = new Tensor[_config.NumLayers];
        }

        private float[,] InitializeEmbeddings(int vocabSize, int dim)
        {
            // Initialize embeddings with small random values
            float[,] embeddings = new float[vocabSize, dim];
            Random rng = new Random(42);
            float scale = (float)Math.Sqrt(1.0 / dim);

            for (int i = 0; i < vocabSize; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    // Normal distribution
                    double u1 = 1.0 - rng.NextDouble();
                    double u2 = 1.0 - rng.NextDouble();
                    double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                    embeddings[i, j] = (float)(normal * scale);
                }
            }

            return embeddings;
        }

        /// <summary>
        /// Forward pass.
        /// </summary>
        /// <param name="tokens">Token indices [batch, seqLen].</param>
        /// <returns>Logits [batch, seqLen, vocabSize].</returns>
        public Tensor Forward(int[,] tokens)
        {
            int batch = tokens.GetLength(0);
            int seqLen = tokens.GetLength(1);

            if (seqLen > _config.MaxSeqLen)
                throw new ArgumentException($"Sequence length {seqLen} exceeds maximum {_config.MaxSeqLen}");

            // Store for backward
            _lastTokens = Flatten2DArray(tokens);

            // Token embedding lookup
            Tensor embedded = EmbeddingLookup(tokens);
            _lastEmbedded = embedded;

            // Add positional encoding
            Tensor afterPosEnc = _posEncoding.Forward(embedded);
            _lastAfterPosEnc = afterPosEnc;

            // Apply embedding dropout
            if (_embeddingDropout != null && _training)
            {
                afterPosEnc = ApplyDropout3D(afterPosEnc, _embeddingDropout);
            }

            // Pass through Transformer blocks
            Tensor hidden = afterPosEnc;
            for (int i = 0; i < _config.NumLayers; i++)
            {
                hidden = _blocks[i].Forward(hidden);
                _blockOutputs[i] = hidden;
            }

            // Final layer normalization
            Tensor normOutput = ApplyLayerNorm3D(hidden, _finalNorm);
            _lastNormOutput = normOutput;

            // Output projection to vocabulary logits
            Tensor logits = ComputeOutputProjection(normOutput);

            return logits;
        }

        /// <summary>
        /// Compute loss (cross-entropy) for language modeling.
        /// </summary>
        /// <param name="logits">Model output logits [batch, seqLen, vocabSize].</param>
        /// <param name="targets">Target token indices [batch, seqLen].</param>
        /// <returns>Average cross-entropy loss.</returns>
        public float ComputeLoss(Tensor logits, int[,] targets)
        {
            int batch = logits.Shape[0];
            int seqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];

            float totalLoss = 0f;
            int count = 0;

            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    // Compute softmax for this position
                    float maxLogit = float.NegativeInfinity;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        if (logits[b, s, v] > maxLogit)
                            maxLogit = logits[b, s, v];
                    }

                    float sumExp = 0f;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        sumExp += (float)Math.Exp(logits[b, s, v] - maxLogit);
                    }

                    // Cross-entropy: -log(softmax[target])
                    int target = targets[b, s];
                    float logProb = logits[b, s, target] - maxLogit - (float)Math.Log(sumExp);
                    totalLoss -= logProb;
                    count++;
                }
            }

            return totalLoss / count;
        }

        /// <summary>
        /// Backward pass.
        /// </summary>
        /// <param name="logits">Model output logits [batch, seqLen, vocabSize].</param>
        /// <param name="targets">Target token indices [batch, seqLen].</param>
        /// <returns>Gradients have been computed and stored in gradient accumulators.</returns>
        public void Backward(Tensor logits, int[,] targets)
        {
            int batch = logits.Shape[0];
            int seqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];

            // Compute gradient of cross-entropy loss w.r.t. logits
            // d_loss/d_logits = softmax(logits) - one_hot(targets)
            Tensor dLogits = ComputeCrossEntropyGradient(logits, targets);

            // Clear embedding gradients
            Array.Clear(_tokenEmbeddingsGrad, 0, _tokenEmbeddingsGrad.Length);
            if (_outputProjectionGrad != null)
                Array.Clear(_outputProjectionGrad, 0, _outputProjectionGrad.Length);

            // Gradient through output projection
            Tensor dNormOutput = OutputProjectionBackward(dLogits, _lastNormOutput);

            // Gradient through final layer norm
            Tensor dHidden = ApplyLayerNormBackward3D(dNormOutput, _finalNorm, _blockOutputs[_config.NumLayers - 1]);

            // Gradient through Transformer blocks (reverse order)
            for (int i = _config.NumLayers - 1; i >= 0; i--)
            {
                Tensor blockInput = i == 0 ? _lastAfterPosEnc : _blockOutputs[i - 1];

                // Need to re-run forward to set up cached values
                if (i > 0)
                {
                    Tensor prevHidden = _lastAfterPosEnc;
                    for (int j = 0; j < i; j++)
                    {
                        prevHidden = _blocks[j].Forward(prevHidden);
                    }
                }
                _blocks[i].Forward(blockInput);
                dHidden = _blocks[i].Backward(dHidden);
            }

            // Gradient through embedding dropout
            if (_embeddingDropout != null && _training)
            {
                dHidden = ApplyDropoutBackward3D(dHidden, _embeddingDropout);
            }

            // Gradient through positional encoding (just passes through)
            Tensor dEmbedded = _posEncoding.Backward(dHidden);

            // Gradient through token embeddings
            EmbeddingBackward(dEmbedded);
        }

        private const int PARALLEL_THRESHOLD = 64;

        private Tensor EmbeddingLookup(int[,] tokens)
        {
            int batch = tokens.GetLength(0);
            int seqLen = tokens.GetLength(1);
            int totalRows = batch * seqLen;
            int embDim = _config.EmbeddingDim;

            Tensor output = new Tensor(new[] { batch, seqLen, embDim });

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, totalRows, row =>
                {
                    int b = row / seqLen;
                    int s = row % seqLen;
                    int tokenId = tokens[b, s];
                    int outputOffset = row * embDim;

                    for (int d = 0; d < embDim; d++)
                    {
                        output.Data[outputOffset + d] = _tokenEmbeddings[tokenId, d];
                    }
                });
            }
            else
            {
                for (int b = 0; b < batch; b++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int tokenId = tokens[b, s];
                        int outputOffset = (b * seqLen + s) * embDim;
                        for (int d = 0; d < embDim; d++)
                        {
                            output.Data[outputOffset + d] = _tokenEmbeddings[tokenId, d];
                        }
                    }
                }
            }

            return output;
        }

        private void EmbeddingBackward(Tensor dEmbedded)
        {
            int batch = dEmbedded.Shape[0];
            int seqLen = dEmbedded.Shape[1];

            for (int i = 0; i < _lastTokens.Length; i++)
            {
                int b = i / seqLen;
                int s = i % seqLen;
                int tokenId = _lastTokens[i];

                for (int d = 0; d < _config.EmbeddingDim; d++)
                {
                    _tokenEmbeddingsGrad[tokenId, d] += dEmbedded[b, s, d];
                }
            }
        }

        private Tensor ComputeOutputProjection(Tensor hidden)
        {
            int batch = hidden.Shape[0];
            int seqLen = hidden.Shape[1];
            int vocabSize = _config.VocabSize;
            int embDim = _config.EmbeddingDim;
            int totalRows = batch * seqLen;

            Tensor logits = new Tensor(new[] { batch, seqLen, vocabSize });

            if (_config.TieEmbeddings)
            {
                // Use embeddings transposed: hidden @ embeddings^T
                if (totalRows >= PARALLEL_THRESHOLD)
                {
                    Parallel.For(0, totalRows, row =>
                    {
                        int hiddenOffset = row * embDim;
                        int logitsOffset = row * vocabSize;

                        for (int v = 0; v < vocabSize; v++)
                        {
                            float sum = 0f;
                            for (int d = 0; d < embDim; d++)
                            {
                                sum += hidden.Data[hiddenOffset + d] * _tokenEmbeddings[v, d];
                            }
                            logits.Data[logitsOffset + v] = sum;
                        }
                    });
                }
                else
                {
                    for (int row = 0; row < totalRows; row++)
                    {
                        int hiddenOffset = row * embDim;
                        int logitsOffset = row * vocabSize;

                        for (int v = 0; v < vocabSize; v++)
                        {
                            float sum = 0f;
                            for (int d = 0; d < embDim; d++)
                            {
                                sum += hidden.Data[hiddenOffset + d] * _tokenEmbeddings[v, d];
                            }
                            logits.Data[logitsOffset + v] = sum;
                        }
                    }
                }
            }
            else
            {
                // Use separate output projection
                if (totalRows >= PARALLEL_THRESHOLD)
                {
                    Parallel.For(0, totalRows, row =>
                    {
                        int hiddenOffset = row * embDim;
                        int logitsOffset = row * vocabSize;

                        for (int v = 0; v < vocabSize; v++)
                        {
                            float sum = 0f;
                            for (int d = 0; d < embDim; d++)
                            {
                                sum += hidden.Data[hiddenOffset + d] * _outputProjection[d, v];
                            }
                            logits.Data[logitsOffset + v] = sum;
                        }
                    });
                }
                else
                {
                    for (int row = 0; row < totalRows; row++)
                    {
                        int hiddenOffset = row * embDim;
                        int logitsOffset = row * vocabSize;

                        for (int v = 0; v < vocabSize; v++)
                        {
                            float sum = 0f;
                            for (int d = 0; d < embDim; d++)
                            {
                                sum += hidden.Data[hiddenOffset + d] * _outputProjection[d, v];
                            }
                            logits.Data[logitsOffset + v] = sum;
                        }
                    }
                }
            }

            return logits;
        }

        private Tensor OutputProjectionBackward(Tensor dLogits, Tensor hidden)
        {
            int batch = dLogits.Shape[0];
            int seqLen = dLogits.Shape[1];
            int vocabSize = _config.VocabSize;
            int embDim = _config.EmbeddingDim;
            int totalRows = batch * seqLen;

            Tensor dHidden = new Tensor(hidden.Shape);

            if (_config.TieEmbeddings)
            {
                // Gradient flows to both hidden and embeddings
                if (totalRows >= PARALLEL_THRESHOLD)
                {
                    // Thread-local gradient accumulators
                    var localEmbGrads = new System.Threading.ThreadLocal<float[,]>(
                        () => new float[vocabSize, embDim], trackAllValues: true);

                    Parallel.For(0, totalRows, row =>
                    {
                        int logitsOffset = row * vocabSize;
                        int hiddenOffset = row * embDim;
                        var localGrad = localEmbGrads.Value;

                        for (int d = 0; d < embDim; d++)
                        {
                            float sum = 0f;
                            float hiddenVal = hidden.Data[hiddenOffset + d];
                            for (int v = 0; v < vocabSize; v++)
                            {
                                float dLogitVal = dLogits.Data[logitsOffset + v];
                                sum += dLogitVal * _tokenEmbeddings[v, d];
                                localGrad[v, d] += dLogitVal * hiddenVal;
                            }
                            dHidden.Data[hiddenOffset + d] = sum;
                        }
                    });

                    // Aggregate thread-local gradients
                    foreach (var localGrad in localEmbGrads.Values)
                    {
                        for (int v = 0; v < vocabSize; v++)
                        {
                            for (int d = 0; d < embDim; d++)
                            {
                                _tokenEmbeddingsGrad[v, d] += localGrad[v, d];
                            }
                        }
                    }
                    localEmbGrads.Dispose();
                }
                else
                {
                    for (int row = 0; row < totalRows; row++)
                    {
                        int logitsOffset = row * vocabSize;
                        int hiddenOffset = row * embDim;

                        for (int d = 0; d < embDim; d++)
                        {
                            float sum = 0f;
                            float hiddenVal = hidden.Data[hiddenOffset + d];
                            for (int v = 0; v < vocabSize; v++)
                            {
                                float dLogitVal = dLogits.Data[logitsOffset + v];
                                sum += dLogitVal * _tokenEmbeddings[v, d];
                                _tokenEmbeddingsGrad[v, d] += dLogitVal * hiddenVal;
                            }
                            dHidden.Data[hiddenOffset + d] = sum;
                        }
                    }
                }
            }
            else
            {
                if (totalRows >= PARALLEL_THRESHOLD)
                {
                    // Thread-local gradient accumulators
                    var localProjGrads = new System.Threading.ThreadLocal<float[,]>(
                        () => new float[embDim, vocabSize], trackAllValues: true);

                    Parallel.For(0, totalRows, row =>
                    {
                        int logitsOffset = row * vocabSize;
                        int hiddenOffset = row * embDim;
                        var localGrad = localProjGrads.Value;

                        for (int d = 0; d < embDim; d++)
                        {
                            float sum = 0f;
                            float hiddenVal = hidden.Data[hiddenOffset + d];
                            for (int v = 0; v < vocabSize; v++)
                            {
                                float dLogitVal = dLogits.Data[logitsOffset + v];
                                sum += dLogitVal * _outputProjection[d, v];
                                localGrad[d, v] += dLogitVal * hiddenVal;
                            }
                            dHidden.Data[hiddenOffset + d] = sum;
                        }
                    });

                    // Aggregate thread-local gradients
                    foreach (var localGrad in localProjGrads.Values)
                    {
                        for (int d = 0; d < embDim; d++)
                        {
                            for (int v = 0; v < vocabSize; v++)
                            {
                                _outputProjectionGrad[d, v] += localGrad[d, v];
                            }
                        }
                    }
                    localProjGrads.Dispose();
                }
                else
                {
                    for (int row = 0; row < totalRows; row++)
                    {
                        int logitsOffset = row * vocabSize;
                        int hiddenOffset = row * embDim;

                        for (int d = 0; d < embDim; d++)
                        {
                            float sum = 0f;
                            float hiddenVal = hidden.Data[hiddenOffset + d];
                            for (int v = 0; v < vocabSize; v++)
                            {
                                float dLogitVal = dLogits.Data[logitsOffset + v];
                                sum += dLogitVal * _outputProjection[d, v];
                                _outputProjectionGrad[d, v] += dLogitVal * hiddenVal;
                            }
                            dHidden.Data[hiddenOffset + d] = sum;
                        }
                    }
                }
            }

            return dHidden;
        }

        private Tensor ComputeCrossEntropyGradient(Tensor logits, int[,] targets)
        {
            int batch = logits.Shape[0];
            int seqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];
            int totalRows = batch * seqLen;
            float scale = 1.0f / totalRows;

            Tensor gradient = new Tensor(logits.Shape);

            if (totalRows >= PARALLEL_THRESHOLD)
            {
                Parallel.For(0, totalRows, row =>
                {
                    int b = row / seqLen;
                    int s = row % seqLen;
                    int logitsOffset = row * vocabSize;

                    // Compute softmax
                    float maxLogit = float.NegativeInfinity;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        float val = logits.Data[logitsOffset + v];
                        if (val > maxLogit)
                            maxLogit = val;
                    }

                    float sumExp = 0f;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        sumExp += MathF.Exp(logits.Data[logitsOffset + v] - maxLogit);
                    }

                    int target = targets[b, s];
                    for (int v = 0; v < vocabSize; v++)
                    {
                        float softmax = MathF.Exp(logits.Data[logitsOffset + v] - maxLogit) / sumExp;
                        gradient.Data[logitsOffset + v] = (softmax - (v == target ? 1f : 0f)) * scale;
                    }
                });
            }
            else
            {
                for (int row = 0; row < totalRows; row++)
                {
                    int b = row / seqLen;
                    int s = row % seqLen;
                    int logitsOffset = row * vocabSize;

                    // Compute softmax
                    float maxLogit = float.NegativeInfinity;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        float val = logits.Data[logitsOffset + v];
                        if (val > maxLogit)
                            maxLogit = val;
                    }

                    float sumExp = 0f;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        sumExp += MathF.Exp(logits.Data[logitsOffset + v] - maxLogit);
                    }

                    int target = targets[b, s];
                    for (int v = 0; v < vocabSize; v++)
                    {
                        float softmax = MathF.Exp(logits.Data[logitsOffset + v] - maxLogit) / sumExp;
                        gradient.Data[logitsOffset + v] = (softmax - (v == target ? 1f : 0f)) * scale;
                    }
                }
            }

            return gradient;
        }

        private Tensor ApplyLayerNorm3D(Tensor input, LayerNorm layerNorm)
        {
            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int dim = input.Shape[2];

            float[,] input2D = new float[batch * seqLen, dim];
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        input2D[b * seqLen + s, d] = input[b, s, d];
                    }
                }
            }

            if (!layerNorm.Built)
                layerNorm.Build(new[] { batch * seqLen, dim });

            float[,] output2D = layerNorm.Call(input2D);

            Tensor output = new Tensor(input.Shape);
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        output[b, s, d] = output2D[b * seqLen + s, d];
                    }
                }
            }

            return output;
        }

        private Tensor ApplyLayerNormBackward3D(Tensor gradOutput, LayerNorm layerNorm, Tensor originalInput)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];
            int dim = gradOutput.Shape[2];

            // Re-run forward to set up cached values
            float[,] input2D = new float[batch * seqLen, dim];
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        input2D[b * seqLen + s, d] = originalInput[b, s, d];
                    }
                }
            }
            layerNorm.Call(input2D);

            float[,] grad2D = new float[batch * seqLen, dim];
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        grad2D[b * seqLen + s, d] = gradOutput[b, s, d];
                    }
                }
            }

            float[,] dInput2D = layerNorm.Backward(grad2D);

            Tensor dInput = new Tensor(gradOutput.Shape);
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        dInput[b, s, d] = dInput2D[b * seqLen + s, d];
                    }
                }
            }

            return dInput;
        }

        private Tensor ApplyDropout3D(Tensor input, Dropout dropout)
        {
            int batch = input.Shape[0];
            int seqLen = input.Shape[1];
            int dim = input.Shape[2];

            float[,] input2D = new float[batch * seqLen, dim];
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        input2D[b * seqLen + s, d] = input[b, s, d];
                    }
                }
            }

            if (!dropout.Built)
                dropout.Build(new[] { batch * seqLen, dim });

            float[,] dropped = dropout.Call(input2D);

            Tensor output = new Tensor(input.Shape);
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        output[b, s, d] = dropped[b * seqLen + s, d];
                    }
                }
            }

            return output;
        }

        private Tensor ApplyDropoutBackward3D(Tensor gradOutput, Dropout dropout)
        {
            int batch = gradOutput.Shape[0];
            int seqLen = gradOutput.Shape[1];
            int dim = gradOutput.Shape[2];

            float[,] grad2D = new float[batch * seqLen, dim];
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        grad2D[b * seqLen + s, d] = gradOutput[b, s, d];
                    }
                }
            }

            float[,] dInput2D = dropout.Backward(grad2D);

            Tensor dInput = new Tensor(gradOutput.Shape);
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        dInput[b, s, d] = dInput2D[b * seqLen + s, d];
                    }
                }
            }

            return dInput;
        }

        private int[] Flatten2DArray(int[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            int[] flat = new int[rows * cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    flat[i * cols + j] = array[i, j];
                }
            }

            return flat;
        }

        /// <summary>
        /// Get all trainable parameters for optimizer.
        /// </summary>
        public List<(string name, float[,] weights, float[,] gradients)> GetParameters()
        {
            var parameters = new List<(string, float[,], float[,])>();

            // Token embeddings
            parameters.Add(("token_embeddings", _tokenEmbeddings, _tokenEmbeddingsGrad));

            // Transformer blocks
            for (int i = 0; i < _blocks.Length; i++)
            {
                var block = _blocks[i];

                // Attention weights
                parameters.Add(($"block_{i}.attention.wQ", block.Attention.WQ, block.Attention.WQGrad));
                parameters.Add(($"block_{i}.attention.wK", block.Attention.WK, block.Attention.WKGrad));
                parameters.Add(($"block_{i}.attention.wV", block.Attention.WV, block.Attention.WVGrad));
                parameters.Add(($"block_{i}.attention.wO", block.Attention.WO, block.Attention.WOGrad));

                // FFN weights
                parameters.Add(($"block_{i}.ffn.w1", block.FeedForwardLayer.W1, block.FeedForwardLayer.W1Grad));
                parameters.Add(($"block_{i}.ffn.w2", block.FeedForwardLayer.W2, block.FeedForwardLayer.W2Grad));
            }

            // Output projection (if not tied)
            if (!_config.TieEmbeddings && _outputProjection != null)
            {
                parameters.Add(("output_projection", _outputProjection, _outputProjectionGrad));
            }

            return parameters;
        }

        /// <summary>
        /// Count total trainable parameters.
        /// </summary>
        public long CountParameters()
        {
            long total = 0;

            // Token embeddings
            total += _tokenEmbeddings.Length;

            // Transformer blocks
            foreach (var block in _blocks)
            {
                // Attention (4 weight matrices + biases)
                total += block.Attention.WQ.Length;
                total += block.Attention.WK.Length;
                total += block.Attention.WV.Length;
                total += block.Attention.WO.Length;

                // FFN (2 weight matrices + biases)
                total += block.FeedForwardLayer.W1.Length;
                total += block.FeedForwardLayer.W2.Length;

                // LayerNorm parameters (gamma, beta)
                total += 2 * _config.EmbeddingDim; // attn layer norm
                total += 2 * _config.EmbeddingDim; // ffn layer norm
            }

            // Final layer norm
            total += 2 * _config.EmbeddingDim;

            // Output projection
            if (!_config.TieEmbeddings && _outputProjection != null)
            {
                total += _outputProjection.Length;
            }

            return total;
        }
    }
}
