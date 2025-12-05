using System;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Inference
{
    /// <summary>
    /// Key-Value cache for efficient autoregressive generation.
    ///
    /// During text generation, we process tokens one at a time. Without caching,
    /// we'd recompute K and V for all previous tokens every step. With KV-cache,
    /// we store previously computed K and V tensors and only compute for new tokens.
    ///
    /// This reduces generation complexity from O(n²) to O(n) per token.
    /// </summary>
    public class KVCache
    {
        private readonly int _numLayers;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _maxSeqLen;

        // Cache storage: [layer][head] -> (K, V) tensors
        // Each K/V tensor: [batch, cached_seq_len, head_dim]
        private Tensor[,] _keyCache;
        private Tensor[,] _valueCache;

        private int _batchSize;
        private int _cachedLength;

        /// <summary>
        /// Current number of cached positions.
        /// </summary>
        public int CachedLength => _cachedLength;

        /// <summary>
        /// Whether the cache has been initialized.
        /// </summary>
        public bool IsInitialized => _keyCache != null;

        /// <summary>
        /// Create a KV cache.
        /// </summary>
        /// <param name="numLayers">Number of transformer layers.</param>
        /// <param name="numHeads">Number of attention heads per layer.</param>
        /// <param name="headDim">Dimension of each attention head.</param>
        /// <param name="maxSeqLen">Maximum sequence length to cache.</param>
        public KVCache(int numLayers, int numHeads, int headDim, int maxSeqLen)
        {
            _numLayers = numLayers;
            _numHeads = numHeads;
            _headDim = headDim;
            _maxSeqLen = maxSeqLen;
            _cachedLength = 0;
        }

        /// <summary>
        /// Initialize the cache for a given batch size.
        /// </summary>
        public void Initialize(int batchSize)
        {
            _batchSize = batchSize;
            _cachedLength = 0;

            _keyCache = new Tensor[_numLayers, _numHeads];
            _valueCache = new Tensor[_numLayers, _numHeads];

            // Pre-allocate cache tensors
            for (int layer = 0; layer < _numLayers; layer++)
            {
                for (int head = 0; head < _numHeads; head++)
                {
                    _keyCache[layer, head] = new Tensor(new[] { batchSize, _maxSeqLen, _headDim });
                    _valueCache[layer, head] = new Tensor(new[] { batchSize, _maxSeqLen, _headDim });
                }
            }
        }

        /// <summary>
        /// Reset the cache (for starting a new generation).
        /// </summary>
        public void Reset()
        {
            _cachedLength = 0;

            if (_keyCache != null)
            {
                for (int layer = 0; layer < _numLayers; layer++)
                {
                    for (int head = 0; head < _numHeads; head++)
                    {
                        Array.Clear(_keyCache[layer, head].Data, 0, _keyCache[layer, head].Size);
                        Array.Clear(_valueCache[layer, head].Data, 0, _valueCache[layer, head].Size);
                    }
                }
            }
        }

        /// <summary>
        /// Update cache with new K and V values for a layer/head.
        /// </summary>
        /// <param name="layer">Layer index.</param>
        /// <param name="head">Head index.</param>
        /// <param name="newK">New key tensor [batch, new_seq_len, head_dim].</param>
        /// <param name="newV">New value tensor [batch, new_seq_len, head_dim].</param>
        public void Update(int layer, int head, Tensor newK, Tensor newV)
        {
            if (!IsInitialized)
                throw new InvalidOperationException("Cache not initialized. Call Initialize first.");

            int newLen = newK.Shape[1];

            if (_cachedLength + newLen > _maxSeqLen)
                throw new InvalidOperationException($"Cache overflow: {_cachedLength} + {newLen} > {_maxSeqLen}");

            // Copy new K and V into cache at current position
            var kCache = _keyCache[layer, head];
            var vCache = _valueCache[layer, head];

            for (int b = 0; b < _batchSize; b++)
            {
                for (int s = 0; s < newLen; s++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        kCache[b, _cachedLength + s, d] = newK[b, s, d];
                        vCache[b, _cachedLength + s, d] = newV[b, s, d];
                    }
                }
            }
        }

        /// <summary>
        /// Increment the cached length after all layers have been updated.
        /// Call this once per generation step after all Update calls.
        /// </summary>
        public void IncrementLength(int numNewTokens = 1)
        {
            _cachedLength += numNewTokens;
        }

        /// <summary>
        /// Get cached K and V for a layer/head, including new values.
        /// Returns the full K/V tensors up to current cached length + new tokens.
        /// </summary>
        /// <param name="layer">Layer index.</param>
        /// <param name="head">Head index.</param>
        /// <param name="newK">New key tensor for current position(s).</param>
        /// <param name="newV">New value tensor for current position(s).</param>
        /// <returns>Full K and V tensors including cached and new values.</returns>
        public (Tensor fullK, Tensor fullV) GetWithNew(int layer, int head, Tensor newK, Tensor newV)
        {
            if (!IsInitialized)
                throw new InvalidOperationException("Cache not initialized");

            int newLen = newK.Shape[1];
            int totalLen = _cachedLength + newLen;

            // Create output tensors
            var fullK = new Tensor(new[] { _batchSize, totalLen, _headDim });
            var fullV = new Tensor(new[] { _batchSize, totalLen, _headDim });

            var kCache = _keyCache[layer, head];
            var vCache = _valueCache[layer, head];

            // Copy cached values
            for (int b = 0; b < _batchSize; b++)
            {
                for (int s = 0; s < _cachedLength; s++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        fullK[b, s, d] = kCache[b, s, d];
                        fullV[b, s, d] = vCache[b, s, d];
                    }
                }

                // Copy new values
                for (int s = 0; s < newLen; s++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        fullK[b, _cachedLength + s, d] = newK[b, s, d];
                        fullV[b, _cachedLength + s, d] = newV[b, s, d];
                    }
                }
            }

            return (fullK, fullV);
        }

        /// <summary>
        /// Get the full cached K tensor for a layer/head.
        /// </summary>
        public Tensor GetCachedK(int layer, int head)
        {
            if (!IsInitialized || _cachedLength == 0)
                return null;

            var result = new Tensor(new[] { _batchSize, _cachedLength, _headDim });
            var cache = _keyCache[layer, head];

            for (int b = 0; b < _batchSize; b++)
            {
                for (int s = 0; s < _cachedLength; s++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        result[b, s, d] = cache[b, s, d];
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Get the full cached V tensor for a layer/head.
        /// </summary>
        public Tensor GetCachedV(int layer, int head)
        {
            if (!IsInitialized || _cachedLength == 0)
                return null;

            var result = new Tensor(new[] { _batchSize, _cachedLength, _headDim });
            var cache = _valueCache[layer, head];

            for (int b = 0; b < _batchSize; b++)
            {
                for (int s = 0; s < _cachedLength; s++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        result[b, s, d] = cache[b, s, d];
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Get memory usage statistics.
        /// </summary>
        public string GetStats()
        {
            long totalBytes = 0;
            if (_keyCache != null)
            {
                totalBytes = 2L * _numLayers * _numHeads * _batchSize * _maxSeqLen * _headDim * sizeof(float);
            }

            return $"KVCache: {_numLayers} layers, {_numHeads} heads, " +
                   $"cached={_cachedLength}/{_maxSeqLen}, " +
                   $"memory={totalBytes / (1024 * 1024):F1}MB";
        }
    }

    /// <summary>
    /// Manages KV cache for all layers in a Transformer model.
    /// </summary>
    public class TransformerKVCache
    {
        private readonly KVCache _cache;
        private readonly int _numLayers;
        private readonly int _numHeads;

        public int CachedLength => _cache.CachedLength;
        public bool IsInitialized => _cache.IsInitialized;

        public TransformerKVCache(int numLayers, int numHeads, int headDim, int maxSeqLen)
        {
            _numLayers = numLayers;
            _numHeads = numHeads;
            _cache = new KVCache(numLayers, numHeads, headDim, maxSeqLen);
        }

        public void Initialize(int batchSize) => _cache.Initialize(batchSize);
        public void Reset() => _cache.Reset();
        public void IncrementLength(int numNewTokens = 1) => _cache.IncrementLength(numNewTokens);

        /// <summary>
        /// Update cache for a specific layer with K/V tensors for all heads.
        /// </summary>
        /// <param name="layer">Layer index.</param>
        /// <param name="keys">Array of K tensors, one per head.</param>
        /// <param name="values">Array of V tensors, one per head.</param>
        public void UpdateLayer(int layer, Tensor[] keys, Tensor[] values)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                _cache.Update(layer, h, keys[h], values[h]);
            }
        }

        /// <summary>
        /// Get cached and new K/V for all heads in a layer.
        /// </summary>
        public (Tensor[] fullKeys, Tensor[] fullValues) GetLayerWithNew(int layer, Tensor[] newKeys, Tensor[] newValues)
        {
            var fullKeys = new Tensor[_numHeads];
            var fullValues = new Tensor[_numHeads];

            for (int h = 0; h < _numHeads; h++)
            {
                var (k, v) = _cache.GetWithNew(layer, h, newKeys[h], newValues[h]);
                fullKeys[h] = k;
                fullValues[h] = v;
            }

            return (fullKeys, fullValues);
        }

        public string GetStats() => _cache.GetStats();
    }
}
