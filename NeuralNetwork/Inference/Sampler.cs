using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Tensors;

namespace NeuralNetwork.Inference
{
    /// <summary>
    /// Sampling strategies for text generation.
    /// Converts model logits to token selections using various sampling methods.
    /// </summary>
    public class Sampler
    {
        private readonly Random _random;

        /// <summary>
        /// Temperature for softmax. Higher = more random, lower = more deterministic.
        /// </summary>
        public float Temperature { get; set; } = 1.0f;

        /// <summary>
        /// Top-k sampling: only consider top k tokens.
        /// Set to 0 to disable.
        /// </summary>
        public int TopK { get; set; } = 0;

        /// <summary>
        /// Top-p (nucleus) sampling: consider tokens until cumulative probability >= p.
        /// Set to 1.0 to disable.
        /// </summary>
        public float TopP { get; set; } = 1.0f;

        /// <summary>
        /// Repetition penalty: penalize tokens that have appeared recently.
        /// Set to 1.0 to disable.
        /// </summary>
        public float RepetitionPenalty { get; set; } = 1.0f;

        /// <summary>
        /// Number of recent tokens to apply repetition penalty to.
        /// </summary>
        public int RepetitionWindow { get; set; } = 64;

        /// <summary>
        /// Create a sampler with default settings.
        /// </summary>
        public Sampler(int? seed = null)
        {
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
        }

        /// <summary>
        /// Sample a single token from logits.
        /// </summary>
        /// <param name="logits">Logits array of shape [vocab_size].</param>
        /// <param name="recentTokens">Optional list of recent tokens for repetition penalty.</param>
        /// <returns>Sampled token index.</returns>
        public int Sample(float[] logits, List<int> recentTokens = null)
        {
            var processedLogits = (float[])logits.Clone();

            // Apply repetition penalty
            if (RepetitionPenalty != 1.0f && recentTokens != null)
            {
                ApplyRepetitionPenalty(processedLogits, recentTokens);
            }

            // Apply temperature
            if (Temperature != 1.0f)
            {
                ApplyTemperature(processedLogits, Temperature);
            }

            // Convert to probabilities
            var probs = Softmax(processedLogits);

            // Apply top-k filtering
            if (TopK > 0 && TopK < probs.Length)
            {
                probs = ApplyTopK(probs, TopK);
            }

            // Apply top-p (nucleus) filtering
            if (TopP < 1.0f)
            {
                probs = ApplyTopP(probs, TopP);
            }

            // Renormalize
            float sum = probs.Sum();
            if (sum > 0)
            {
                for (int i = 0; i < probs.Length; i++)
                    probs[i] /= sum;
            }

            // Sample from distribution
            return SampleFromDistribution(probs);
        }

        /// <summary>
        /// Sample from model output tensor.
        /// </summary>
        /// <param name="logits">Logits tensor [batch, seq_len, vocab_size].</param>
        /// <param name="position">Sequence position to sample from (-1 for last).</param>
        /// <param name="batchIndex">Batch index to sample from.</param>
        /// <param name="recentTokens">Recent tokens for repetition penalty.</param>
        /// <returns>Sampled token index.</returns>
        public int SampleFromTensor(Tensor logits, int position = -1, int batchIndex = 0, List<int> recentTokens = null)
        {
            int seqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];

            if (position < 0)
                position = seqLen + position;

            // Extract logits for the specified position
            var positionLogits = new float[vocabSize];
            for (int v = 0; v < vocabSize; v++)
            {
                positionLogits[v] = logits[batchIndex, position, v];
            }

            return Sample(positionLogits, recentTokens);
        }

        /// <summary>
        /// Greedy sampling: always select the highest probability token.
        /// </summary>
        public int SampleGreedy(float[] logits)
        {
            int bestIdx = 0;
            float bestLogit = logits[0];

            for (int i = 1; i < logits.Length; i++)
            {
                if (logits[i] > bestLogit)
                {
                    bestLogit = logits[i];
                    bestIdx = i;
                }
            }

            return bestIdx;
        }

        /// <summary>
        /// Greedy sampling from tensor.
        /// </summary>
        public int SampleGreedyFromTensor(Tensor logits, int position = -1, int batchIndex = 0)
        {
            int seqLen = logits.Shape[1];
            int vocabSize = logits.Shape[2];

            if (position < 0)
                position = seqLen + position;

            int bestIdx = 0;
            float bestLogit = logits[batchIndex, position, 0];

            for (int v = 1; v < vocabSize; v++)
            {
                if (logits[batchIndex, position, v] > bestLogit)
                {
                    bestLogit = logits[batchIndex, position, v];
                    bestIdx = v;
                }
            }

            return bestIdx;
        }

        #region Sampling Methods

        private void ApplyTemperature(float[] logits, float temperature)
        {
            for (int i = 0; i < logits.Length; i++)
            {
                logits[i] /= temperature;
            }
        }

        private void ApplyRepetitionPenalty(float[] logits, List<int> recentTokens)
        {
            // Get unique recent tokens within window
            var windowTokens = recentTokens
                .Skip(Math.Max(0, recentTokens.Count - RepetitionWindow))
                .Distinct()
                .ToHashSet();

            foreach (var token in windowTokens)
            {
                if (token >= 0 && token < logits.Length)
                {
                    // If logit is positive, divide by penalty; if negative, multiply
                    if (logits[token] > 0)
                        logits[token] /= RepetitionPenalty;
                    else
                        logits[token] *= RepetitionPenalty;
                }
            }
        }

        private float[] Softmax(float[] logits)
        {
            float maxLogit = logits.Max();
            var exp = logits.Select(x => (float)Math.Exp(x - maxLogit)).ToArray();
            float sum = exp.Sum();
            return exp.Select(x => x / sum).ToArray();
        }

        private float[] ApplyTopK(float[] probs, int k)
        {
            // Get indices of top-k probabilities
            var indexed = probs
                .Select((p, i) => (prob: p, index: i))
                .OrderByDescending(x => x.prob)
                .Take(k)
                .ToHashSet();

            var topKIndices = indexed.Select(x => x.index).ToHashSet();

            // Zero out non-top-k probabilities
            var result = new float[probs.Length];
            for (int i = 0; i < probs.Length; i++)
            {
                result[i] = topKIndices.Contains(i) ? probs[i] : 0f;
            }

            return result;
        }

        private float[] ApplyTopP(float[] probs, float p)
        {
            // Sort indices by probability descending
            var sorted = probs
                .Select((prob, idx) => (prob, idx))
                .OrderByDescending(x => x.prob)
                .ToList();

            // Find cutoff where cumulative probability exceeds p
            float cumulative = 0f;
            var keepIndices = new HashSet<int>();

            foreach (var (prob, idx) in sorted)
            {
                keepIndices.Add(idx);
                cumulative += prob;
                if (cumulative >= p)
                    break;
            }

            // Zero out indices not in nucleus
            var result = new float[probs.Length];
            for (int i = 0; i < probs.Length; i++)
            {
                result[i] = keepIndices.Contains(i) ? probs[i] : 0f;
            }

            return result;
        }

        private int SampleFromDistribution(float[] probs)
        {
            float r = (float)_random.NextDouble();
            float cumulative = 0f;

            for (int i = 0; i < probs.Length; i++)
            {
                cumulative += probs[i];
                if (r < cumulative)
                    return i;
            }

            // Fallback: return last non-zero index or 0
            for (int i = probs.Length - 1; i >= 0; i--)
            {
                if (probs[i] > 0)
                    return i;
            }

            return 0;
        }

        #endregion

        #region Factory Methods

        /// <summary>
        /// Create a greedy sampler (temperature = 0 effectively).
        /// </summary>
        public static Sampler Greedy()
        {
            return new Sampler { Temperature = 0.01f, TopK = 1 };
        }

        /// <summary>
        /// Create a sampler for creative text generation.
        /// </summary>
        public static Sampler Creative(int? seed = null)
        {
            return new Sampler(seed)
            {
                Temperature = 0.9f,
                TopP = 0.95f,
                TopK = 50,
                RepetitionPenalty = 1.1f
            };
        }

        /// <summary>
        /// Create a balanced sampler for general use.
        /// </summary>
        public static Sampler Balanced(int? seed = null)
        {
            return new Sampler(seed)
            {
                Temperature = 0.7f,
                TopP = 0.9f,
                TopK = 40,
                RepetitionPenalty = 1.05f
            };
        }

        /// <summary>
        /// Create a conservative sampler for factual text.
        /// </summary>
        public static Sampler Conservative(int? seed = null)
        {
            return new Sampler(seed)
            {
                Temperature = 0.3f,
                TopP = 0.8f,
                TopK = 20,
                RepetitionPenalty = 1.0f
            };
        }

        #endregion

        public override string ToString()
        {
            return $"Sampler(temp={Temperature}, top_k={TopK}, top_p={TopP}, rep_penalty={RepetitionPenalty})";
        }
    }

    /// <summary>
    /// Beam search implementation for higher quality generation.
    /// </summary>
    public class BeamSearch
    {
        private readonly int _beamWidth;
        private readonly float _lengthPenalty;
        private readonly int _maxLength;

        public BeamSearch(int beamWidth = 4, float lengthPenalty = 0.6f, int maxLength = 100)
        {
            _beamWidth = beamWidth;
            _lengthPenalty = lengthPenalty;
            _maxLength = maxLength;
        }

        /// <summary>
        /// Beam search result: sequence of tokens and score.
        /// </summary>
        public class BeamResult
        {
            public List<int> Tokens { get; set; }
            public float Score { get; set; }
            public bool Finished { get; set; }
        }

        /// <summary>
        /// Run beam search on logits from a single step.
        /// Note: Full beam search requires model integration. This is a step helper.
        /// </summary>
        /// <param name="beams">Current beam candidates.</param>
        /// <param name="logits">Logits for current step [beam_width, vocab_size].</param>
        /// <param name="eosToken">End of sequence token ID.</param>
        /// <returns>Updated beam candidates.</returns>
        public List<BeamResult> Step(List<BeamResult> beams, float[,] logits, int eosToken)
        {
            int vocabSize = logits.GetLength(1);
            var candidates = new List<(float score, List<int> tokens, bool finished)>();

            for (int b = 0; b < beams.Count; b++)
            {
                if (beams[b].Finished)
                {
                    // Keep finished beams as-is
                    candidates.Add((beams[b].Score, beams[b].Tokens, true));
                    continue;
                }

                // Get log probabilities for this beam
                var logProbs = new float[vocabSize];
                float maxLogit = float.NegativeInfinity;
                for (int v = 0; v < vocabSize; v++)
                    maxLogit = Math.Max(maxLogit, logits[b, v]);

                float sumExp = 0f;
                for (int v = 0; v < vocabSize; v++)
                    sumExp += (float)Math.Exp(logits[b, v] - maxLogit);

                float logSumExp = maxLogit + (float)Math.Log(sumExp);
                for (int v = 0; v < vocabSize; v++)
                    logProbs[v] = logits[b, v] - logSumExp;

                // Expand beam with top-k tokens
                var topK = logProbs
                    .Select((lp, idx) => (logProb: lp, token: idx))
                    .OrderByDescending(x => x.logProb)
                    .Take(_beamWidth * 2)
                    .ToList();

                foreach (var (logProb, token) in topK)
                {
                    var newTokens = new List<int>(beams[b].Tokens) { token };
                    float newScore = beams[b].Score + logProb;

                    // Apply length penalty
                    float lengthNorm = (float)Math.Pow((5 + newTokens.Count) / 6.0, _lengthPenalty);
                    float normalizedScore = newScore / lengthNorm;

                    bool finished = token == eosToken || newTokens.Count >= _maxLength;

                    candidates.Add((normalizedScore, newTokens, finished));
                }
            }

            // Select top beams
            var topBeams = candidates
                .OrderByDescending(c => c.score)
                .Take(_beamWidth)
                .Select(c => new BeamResult
                {
                    Tokens = c.tokens,
                    Score = c.score,
                    Finished = c.finished
                })
                .ToList();

            return topBeams;
        }

        /// <summary>
        /// Initialize beams for search.
        /// </summary>
        public List<BeamResult> InitializeBeams(List<int> promptTokens)
        {
            return new List<BeamResult>
            {
                new BeamResult
                {
                    Tokens = new List<int>(promptTokens),
                    Score = 0f,
                    Finished = false
                }
            };
        }

        /// <summary>
        /// Check if all beams are finished.
        /// </summary>
        public bool AllFinished(List<BeamResult> beams)
        {
            return beams.All(b => b.Finished);
        }

        /// <summary>
        /// Get the best completed sequence.
        /// </summary>
        public BeamResult GetBest(List<BeamResult> beams)
        {
            return beams
                .Where(b => b.Finished)
                .OrderByDescending(b => b.Score)
                .FirstOrDefault()
                ?? beams.OrderByDescending(b => b.Score).First();
        }
    }
}
