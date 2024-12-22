using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Processing.Text;

namespace NeuralNetwork.Inference
{
    public class TextGenerator
    {
        private Sequential model;
        private Dictionary<int, string> reverseVocab;
        private int maxLen;

        public TextGenerator(Sequential model, Dictionary<int, string> reverseVocab, int maxLen)
        {
            this.model = model;
            this.reverseVocab = reverseVocab;
            this.maxLen = maxLen;
        }

        public string GenerateText(string seedText, int numWords)
        {
            var vocab = reverseVocab.ToDictionary(kv => kv.Value, kv => kv.Key);
            var sequences = TextPreProcessing.TextToSequences(seedText, vocab);
            sequences = TextPreProcessing.PadSequences(sequences, maxLen);

            var generatedText = new List<string>(seedText.Split(' '));
            for (int i = 0; i < numWords; i++)
            {
                var predictions = model.Predict(sequences);
                int predictedIndex = Sample(predictions);
                var predictedWord = reverseVocab[predictedIndex];
                generatedText.Add(predictedWord);

                var newSeq = new float[1, maxLen];
                for (int j = 0; j < maxLen - 1; j++)
                {
                    newSeq[0, j] = sequences[0, j + 1];
                }
                newSeq[0, maxLen - 1] = predictedIndex;
                sequences = newSeq;
            }

            return string.Join(" ", generatedText);
        }

        private int Sample(float[,] predictions)
        {
            Random rnd = new Random();
            float sum = predictions.Cast<float>().Sum();
            float randValue = rnd.NextSingle() * sum;
            float cumulative = 0.0f;
            for (int i = 0; i < predictions.GetLength(1); i++)
            {
                cumulative += predictions[0, i];
                if (randValue < cumulative)
                {
                    return i;
                }
            }
            return predictions.GetLength(1) - 1;
        }
    }
}
