using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Processing.Text
{
    public static class TextPreProcessing
    {
        public static Dictionary<string, int> Tokenize(string text)
        {
            var tokens = text.Split(new char[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            var vocab = tokens.Distinct().Select((token, index) => new { token, index }).ToDictionary(t => t.token, t => t.index);
            return vocab;
        }

        public static float[,] TextToSequences(string text, Dictionary<string, int> vocab)
        {
            var tokens = text.Split(new char[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            float[,] sequences = new float[tokens.Length, 1];

            for (int i = 0; i < tokens.Length; i++)
            {
                if (vocab.ContainsKey(tokens[i]))
                {
                    sequences[i, 0] = vocab[tokens[i]];
                }
                else
                {
                    throw new ArgumentException($"Token '{tokens[i]}' not found in vocabulary.");
                }
            }
            return sequences;
        }

        public static float[,] PadSequences(float[,] sequences, int maxLen)
        {
            int samples = sequences.GetLength(0);
            float[,] paddedSequences = new float[samples, maxLen];
            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < sequences.GetLength(1) && j < maxLen; j++)
                {
                    paddedSequences[i, j] = sequences[i, j];
                }
            }
            return paddedSequences;
        }
    }
}
