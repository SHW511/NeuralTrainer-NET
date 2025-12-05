using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace NeuralNetwork.Processing.Text
{
    /// <summary>
    /// Byte-Pair Encoding (BPE) Tokenizer.
    ///
    /// BPE is a subword tokenization algorithm that:
    /// 1. Starts with a character-level vocabulary
    /// 2. Iteratively merges the most frequent pair of tokens
    /// 3. Results in a vocabulary of subword units
    ///
    /// This allows handling of rare words by breaking them into known subwords,
    /// while keeping common words as single tokens.
    /// </summary>
    public class BPETokenizer
    {
        // Special tokens
        public const string PAD_TOKEN = "<PAD>";
        public const string UNK_TOKEN = "<UNK>";
        public const string BOS_TOKEN = "<BOS>";
        public const string EOS_TOKEN = "<EOS>";

        public int PadId { get; private set; }
        public int UnkId { get; private set; }
        public int BosId { get; private set; }
        public int EosId { get; private set; }

        // Vocabulary mappings
        private Dictionary<string, int> _tokenToId;
        private Dictionary<int, string> _idToToken;

        // BPE merge rules (ordered)
        private List<(string, string)> _merges;

        // Pre-tokenization pattern (similar to GPT-2)
        // Splits on whitespace boundaries while keeping the space attached to the following word
        private static readonly Regex _preTokenizePattern = new Regex(
            @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled);

        // Character used to mark word boundaries (beginning of word)
        private const string WORD_BOUNDARY = "Ġ";  // GPT-2 style (represents space)

        public int VocabSize => _tokenToId?.Count ?? 0;
        public bool IsTrained => _merges != null && _merges.Count > 0;

        public BPETokenizer()
        {
            _tokenToId = new Dictionary<string, int>();
            _idToToken = new Dictionary<int, string>();
            _merges = new List<(string, string)>();
        }

        /// <summary>
        /// Train the BPE tokenizer on a corpus of text.
        /// </summary>
        /// <param name="text">Training text corpus.</param>
        /// <param name="vocabSize">Target vocabulary size (including special tokens).</param>
        /// <param name="minFrequency">Minimum frequency for a pair to be merged.</param>
        public void Train(string text, int vocabSize, int minFrequency = 2)
        {
            Console.WriteLine($"Training BPE tokenizer with target vocab size: {vocabSize}");

            // Initialize with special tokens
            InitializeSpecialTokens();

            // Pre-tokenize the text
            var words = PreTokenize(text);
            Console.WriteLine($"Pre-tokenized into {words.Count} words");

            // Build initial word frequencies
            var wordFreqs = new Dictionary<string, int>();
            foreach (var word in words)
            {
                if (!wordFreqs.ContainsKey(word))
                    wordFreqs[word] = 0;
                wordFreqs[word]++;
            }

            // Split words into characters (initial tokenization)
            // Each word becomes a list of character tokens
            var splits = new Dictionary<string, List<string>>();
            foreach (var word in wordFreqs.Keys)
            {
                splits[word] = word.Select(c => c.ToString()).ToList();
            }

            // Build initial character vocabulary
            var charVocab = new HashSet<string>();
            foreach (var split in splits.Values)
            {
                foreach (var ch in split)
                    charVocab.Add(ch);
            }

            // Add characters to vocabulary
            foreach (var ch in charVocab.OrderBy(c => c))
            {
                if (!_tokenToId.ContainsKey(ch))
                {
                    int id = _tokenToId.Count;
                    _tokenToId[ch] = id;
                    _idToToken[id] = ch;
                }
            }

            Console.WriteLine($"Initial character vocabulary: {_tokenToId.Count} tokens");

            // Iteratively merge most frequent pairs
            int targetMerges = vocabSize - _tokenToId.Count;
            int mergeCount = 0;

            while (_tokenToId.Count < vocabSize)
            {
                // Count pair frequencies
                var pairFreqs = new Dictionary<(string, string), int>();
                foreach (var (word, freq) in wordFreqs)
                {
                    var wordSplit = splits[word];
                    if (wordSplit.Count < 2) continue;

                    for (int i = 0; i < wordSplit.Count - 1; i++)
                    {
                        var pair = (wordSplit[i], wordSplit[i + 1]);
                        if (!pairFreqs.ContainsKey(pair))
                            pairFreqs[pair] = 0;
                        pairFreqs[pair] += freq;
                    }
                }

                if (pairFreqs.Count == 0)
                {
                    Console.WriteLine("No more pairs to merge");
                    break;
                }

                // Find most frequent pair
                var bestPair = pairFreqs.OrderByDescending(p => p.Value).First();
                if (bestPair.Value < minFrequency)
                {
                    Console.WriteLine($"Best pair frequency {bestPair.Value} below minimum {minFrequency}");
                    break;
                }

                var (first, second) = bestPair.Key;
                string merged = first + second;

                // Add merge rule
                _merges.Add((first, second));

                // Add merged token to vocabulary
                if (!_tokenToId.ContainsKey(merged))
                {
                    int id = _tokenToId.Count;
                    _tokenToId[merged] = id;
                    _idToToken[id] = merged;
                }

                // Apply merge to all words
                foreach (var word in splits.Keys.ToList())
                {
                    splits[word] = ApplyMerge(splits[word], first, second, merged);
                }

                mergeCount++;
                if (mergeCount % 100 == 0)
                {
                    Console.WriteLine($"Merge {mergeCount}: '{first}' + '{second}' -> '{merged}' (freq: {bestPair.Value})");
                }
            }

            Console.WriteLine($"Training complete. Final vocabulary size: {VocabSize}");
            Console.WriteLine($"Total merges learned: {_merges.Count}");
        }

        /// <summary>
        /// Train from a text file.
        /// </summary>
        public void TrainFromFile(string filePath, int vocabSize, int minFrequency = 2)
        {
            string text = File.ReadAllText(filePath);
            Train(text, vocabSize, minFrequency);
        }

        /// <summary>
        /// Encode text into token IDs.
        /// </summary>
        /// <param name="text">Text to encode.</param>
        /// <param name="addBos">Whether to prepend BOS token.</param>
        /// <param name="addEos">Whether to append EOS token.</param>
        /// <returns>Array of token IDs.</returns>
        public int[] Encode(string text, bool addBos = false, bool addEos = false)
        {
            if (!IsTrained && VocabSize == 0)
                throw new InvalidOperationException("Tokenizer has not been trained");

            var tokens = new List<int>();

            if (addBos)
                tokens.Add(BosId);

            // Pre-tokenize
            var words = PreTokenize(text);

            foreach (var word in words)
            {
                // Apply BPE to each word
                var wordTokens = EncodeWord(word);
                tokens.AddRange(wordTokens);
            }

            if (addEos)
                tokens.Add(EosId);

            return tokens.ToArray();
        }

        /// <summary>
        /// Encode a batch of texts.
        /// </summary>
        /// <param name="texts">Texts to encode.</param>
        /// <param name="maxLength">Maximum sequence length (pad/truncate to this).</param>
        /// <param name="addBos">Whether to prepend BOS token.</param>
        /// <param name="addEos">Whether to append EOS token.</param>
        /// <returns>2D array [batch, seqLen] of token IDs.</returns>
        public int[,] EncodeBatch(string[] texts, int maxLength, bool addBos = false, bool addEos = false)
        {
            var encoded = texts.Select(t => Encode(t, addBos, addEos)).ToArray();
            return PadSequences(encoded, maxLength);
        }

        /// <summary>
        /// Decode token IDs back to text.
        /// </summary>
        /// <param name="ids">Token IDs to decode.</param>
        /// <param name="skipSpecialTokens">Whether to skip special tokens in output.</param>
        /// <returns>Decoded text.</returns>
        public string Decode(int[] ids, bool skipSpecialTokens = true)
        {
            var sb = new StringBuilder();

            foreach (var id in ids)
            {
                if (!_idToToken.TryGetValue(id, out string token))
                {
                    if (!skipSpecialTokens)
                        sb.Append(UNK_TOKEN);
                    continue;
                }

                // Skip special tokens if requested
                if (skipSpecialTokens && IsSpecialToken(token))
                    continue;

                // Convert word boundary marker back to space
                string decoded = token.Replace(WORD_BOUNDARY, " ");
                sb.Append(decoded);
            }

            return sb.ToString().Trim();
        }

        /// <summary>
        /// Decode a batch of sequences.
        /// </summary>
        public string[] DecodeBatch(int[,] ids, bool skipSpecialTokens = true)
        {
            int batch = ids.GetLength(0);
            int seqLen = ids.GetLength(1);
            var results = new string[batch];

            for (int b = 0; b < batch; b++)
            {
                var sequence = new int[seqLen];
                for (int s = 0; s < seqLen; s++)
                    sequence[s] = ids[b, s];
                results[b] = Decode(sequence, skipSpecialTokens);
            }

            return results;
        }

        /// <summary>
        /// Get the token string for an ID.
        /// </summary>
        public string IdToToken(int id)
        {
            return _idToToken.TryGetValue(id, out var token) ? token : UNK_TOKEN;
        }

        /// <summary>
        /// Get the ID for a token string.
        /// </summary>
        public int TokenToId(string token)
        {
            return _tokenToId.TryGetValue(token, out var id) ? id : UnkId;
        }

        /// <summary>
        /// Save the trained tokenizer to a directory.
        /// </summary>
        public void Save(string directory)
        {
            if (!Directory.Exists(directory))
                Directory.CreateDirectory(directory);

            // Save vocabulary
            var vocabPath = Path.Combine(directory, "vocab.json");
            var vocabJson = JsonSerializer.Serialize(_tokenToId, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(vocabPath, vocabJson);

            // Save merges
            var mergesPath = Path.Combine(directory, "merges.txt");
            var mergeLines = _merges.Select(m => $"{m.Item1} {m.Item2}");
            File.WriteAllLines(mergesPath, mergeLines);

            // Save config
            var configPath = Path.Combine(directory, "tokenizer_config.json");
            var config = new Dictionary<string, object>
            {
                ["vocab_size"] = VocabSize,
                ["pad_token"] = PAD_TOKEN,
                ["unk_token"] = UNK_TOKEN,
                ["bos_token"] = BOS_TOKEN,
                ["eos_token"] = EOS_TOKEN,
                ["pad_id"] = PadId,
                ["unk_id"] = UnkId,
                ["bos_id"] = BosId,
                ["eos_id"] = EosId
            };
            var configJson = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(configPath, configJson);

            Console.WriteLine($"Tokenizer saved to {directory}");
        }

        /// <summary>
        /// Load a trained tokenizer from a directory.
        /// </summary>
        public static BPETokenizer Load(string directory)
        {
            var tokenizer = new BPETokenizer();

            // Load vocabulary
            var vocabPath = Path.Combine(directory, "vocab.json");
            var vocabJson = File.ReadAllText(vocabPath);
            tokenizer._tokenToId = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson);
            tokenizer._idToToken = tokenizer._tokenToId.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            // Load merges
            var mergesPath = Path.Combine(directory, "merges.txt");
            var mergeLines = File.ReadAllLines(mergesPath);
            tokenizer._merges = mergeLines
                .Where(line => !string.IsNullOrWhiteSpace(line))
                .Select(line =>
                {
                    var parts = line.Split(' ');
                    return (parts[0], parts[1]);
                })
                .ToList();

            // Load config for special token IDs
            var configPath = Path.Combine(directory, "tokenizer_config.json");
            if (File.Exists(configPath))
            {
                var configJson = File.ReadAllText(configPath);
                var config = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(configJson);
                tokenizer.PadId = config["pad_id"].GetInt32();
                tokenizer.UnkId = config["unk_id"].GetInt32();
                tokenizer.BosId = config["bos_id"].GetInt32();
                tokenizer.EosId = config["eos_id"].GetInt32();
            }
            else
            {
                // Fallback: look up special tokens in vocab
                tokenizer.PadId = tokenizer._tokenToId.GetValueOrDefault(PAD_TOKEN, 0);
                tokenizer.UnkId = tokenizer._tokenToId.GetValueOrDefault(UNK_TOKEN, 1);
                tokenizer.BosId = tokenizer._tokenToId.GetValueOrDefault(BOS_TOKEN, 2);
                tokenizer.EosId = tokenizer._tokenToId.GetValueOrDefault(EOS_TOKEN, 3);
            }

            Console.WriteLine($"Tokenizer loaded from {directory} (vocab size: {tokenizer.VocabSize})");
            return tokenizer;
        }

        #region Private Methods

        private void InitializeSpecialTokens()
        {
            _tokenToId.Clear();
            _idToToken.Clear();

            // Add special tokens first (fixed positions)
            PadId = 0;
            _tokenToId[PAD_TOKEN] = PadId;
            _idToToken[PadId] = PAD_TOKEN;

            UnkId = 1;
            _tokenToId[UNK_TOKEN] = UnkId;
            _idToToken[UnkId] = UNK_TOKEN;

            BosId = 2;
            _tokenToId[BOS_TOKEN] = BosId;
            _idToToken[BosId] = BOS_TOKEN;

            EosId = 3;
            _tokenToId[EOS_TOKEN] = EosId;
            _idToToken[EosId] = EOS_TOKEN;
        }

        private List<string> PreTokenize(string text)
        {
            var words = new List<string>();
            var matches = _preTokenizePattern.Matches(text);

            foreach (Match match in matches)
            {
                string word = match.Value;

                // Mark word boundaries (leading space becomes Ġ prefix)
                if (word.StartsWith(" "))
                {
                    word = WORD_BOUNDARY + word.Substring(1);
                }

                if (!string.IsNullOrEmpty(word))
                    words.Add(word);
            }

            return words;
        }

        private int[] EncodeWord(string word)
        {
            // Start with character-level split
            var tokens = word.Select(c => c.ToString()).ToList();

            // Apply learned merges in order
            foreach (var (first, second) in _merges)
            {
                tokens = ApplyMerge(tokens, first, second, first + second);
            }

            // Convert tokens to IDs
            var ids = new List<int>();
            foreach (var token in tokens)
            {
                if (_tokenToId.TryGetValue(token, out int id))
                {
                    ids.Add(id);
                }
                else
                {
                    // Unknown token - could try to break down further or use UNK
                    ids.Add(UnkId);
                }
            }

            return ids.ToArray();
        }

        private List<string> ApplyMerge(List<string> tokens, string first, string second, string merged)
        {
            var result = new List<string>();
            int i = 0;

            while (i < tokens.Count)
            {
                if (i < tokens.Count - 1 && tokens[i] == first && tokens[i + 1] == second)
                {
                    result.Add(merged);
                    i += 2;
                }
                else
                {
                    result.Add(tokens[i]);
                    i++;
                }
            }

            return result;
        }

        private bool IsSpecialToken(string token)
        {
            return token == PAD_TOKEN || token == UNK_TOKEN ||
                   token == BOS_TOKEN || token == EOS_TOKEN;
        }

        private int[,] PadSequences(int[][] sequences, int maxLength)
        {
            int batch = sequences.Length;
            var result = new int[batch, maxLength];

            // Initialize with pad tokens
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < maxLength; s++)
                {
                    result[b, s] = PadId;
                }
            }

            // Copy sequences (truncating if necessary)
            for (int b = 0; b < batch; b++)
            {
                int copyLen = Math.Min(sequences[b].Length, maxLength);
                for (int s = 0; s < copyLen; s++)
                {
                    result[b, s] = sequences[b][s];
                }
            }

            return result;
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Create a simple character-level tokenizer (no BPE merges).
        /// Useful for quick testing.
        /// </summary>
        public static BPETokenizer CreateCharacterLevel(string text)
        {
            var tokenizer = new BPETokenizer();
            tokenizer.InitializeSpecialTokens();

            // Add all unique characters
            var chars = text.ToHashSet();
            foreach (var ch in chars.OrderBy(c => c))
            {
                string token = ch.ToString();
                if (!tokenizer._tokenToId.ContainsKey(token))
                {
                    int id = tokenizer._tokenToId.Count;
                    tokenizer._tokenToId[token] = id;
                    tokenizer._idToToken[id] = token;
                }
            }

            // No merges for character-level
            tokenizer._merges = new List<(string, string)>();

            return tokenizer;
        }

        /// <summary>
        /// Get vocabulary statistics.
        /// </summary>
        public string GetStats()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Vocabulary Size: {VocabSize}");
            sb.AppendLine($"Number of Merges: {_merges.Count}");
            sb.AppendLine($"Special Tokens: PAD={PadId}, UNK={UnkId}, BOS={BosId}, EOS={EosId}");

            // Token length distribution
            var lengths = _tokenToId.Keys.Select(t => t.Length).GroupBy(l => l).OrderBy(g => g.Key);
            sb.AppendLine("Token length distribution:");
            foreach (var group in lengths.Take(10))
            {
                sb.AppendLine($"  Length {group.Key}: {group.Count()} tokens");
            }

            return sb.ToString();
        }

        #endregion
    }
}
