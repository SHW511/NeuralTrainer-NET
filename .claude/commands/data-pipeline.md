# Data Pipeline

You are a data engineering specialist for NeuralTrainer-NET, helping to create data loading and preprocessing pipelines.

## Your Task

Create or enhance data loading and preprocessing pipelines. The user will specify: $ARGUMENTS

## Pipeline Components

```
Raw Data → Loading → Preprocessing → Batching → Training
              ↓           ↓            ↓
           Format      Normalize    Shuffle
           Parse       Augment      Pad
           Validate    Tokenize     Cache
```

## Steps to Follow

1. **Understand Data Requirements**
   - Data format (CSV, JSON, binary, images, audio)
   - Size considerations (fits in memory vs streaming)
   - Preprocessing needs
   - Batching strategy

2. **Study Existing Pipelines**
   - Review `NeuralNetwork/Processing/Text/TextPreProcessing.cs`
   - Review `NeuralNetwork/Processing/Text/TextDataset.cs`
   - Review `NeuralNetwork/Processing/Text/BPETokenizer.cs`

3. **Implement Pipeline Components**
   - Create in `NeuralNetwork/Processing/{Modality}/`
   - Follow existing patterns
   - Consider memory efficiency

4. **Test Pipeline**
   - Verify correct data shapes
   - Check preprocessing correctness
   - Measure throughput

## Dataset Interface

```csharp
namespace NeuralNetwork.Processing
{
    public interface IDataset
    {
        int Count { get; }
        int BatchSize { get; set; }
        int NumBatches { get; }

        void Shuffle();
        (float[,] X, float[,] Y) GetBatch(int batchIndex);
        IEnumerable<(float[,] X, float[,] Y)> GetBatches();
    }
}
```

## Implementation Template

```csharp
namespace NeuralNetwork.Processing
{
    public class CustomDataset : IDataset
    {
        private float[,] data;
        private float[,] labels;
        private int[] indices;
        private Random rng = new Random();

        public int Count => data.GetLength(0);
        public int BatchSize { get; set; } = 32;
        public int NumBatches => (Count + BatchSize - 1) / BatchSize;

        public CustomDataset(string dataPath)
        {
            LoadData(dataPath);
            indices = Enumerable.Range(0, Count).ToArray();
        }

        private void LoadData(string path)
        {
            // Load and parse data file
            // Store in data and labels arrays
        }

        public void Shuffle()
        {
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        public (float[,] X, float[,] Y) GetBatch(int batchIndex)
        {
            int start = batchIndex * BatchSize;
            int end = Math.Min(start + BatchSize, Count);
            int actualBatchSize = end - start;

            int inputDim = data.GetLength(1);
            int outputDim = labels.GetLength(1);

            float[,] X = new float[actualBatchSize, inputDim];
            float[,] Y = new float[actualBatchSize, outputDim];

            for (int i = 0; i < actualBatchSize; i++)
            {
                int idx = indices[start + i];
                for (int j = 0; j < inputDim; j++)
                    X[i, j] = data[idx, j];
                for (int j = 0; j < outputDim; j++)
                    Y[i, j] = labels[idx, j];
            }

            return (X, Y);
        }

        public IEnumerable<(float[,] X, float[,] Y)> GetBatches()
        {
            for (int i = 0; i < NumBatches; i++)
                yield return GetBatch(i);
        }
    }
}
```

## Preprocessing Utilities

### Normalization
```csharp
public static class Normalize
{
    // Z-score normalization
    public static float[,] ZScore(float[,] data, out float[] means, out float[] stds)
    {
        int n = data.GetLength(0);
        int m = data.GetLength(1);
        means = new float[m];
        stds = new float[m];

        // Compute means
        for (int j = 0; j < m; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += data[i, j];
            means[j] /= n;
        }

        // Compute stds
        for (int j = 0; j < m; j++)
        {
            for (int i = 0; i < n; i++)
                stds[j] += (data[i, j] - means[j]) * (data[i, j] - means[j]);
            stds[j] = MathF.Sqrt(stds[j] / n) + 1e-8f;
        }

        // Normalize
        float[,] result = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                result[i, j] = (data[i, j] - means[j]) / stds[j];

        return result;
    }

    // Min-max normalization
    public static float[,] MinMax(float[,] data, float min = 0f, float max = 1f)
    {
        // Implementation
    }
}
```

### Data Augmentation
```csharp
public static class Augmentation
{
    // For image data
    public static float[,,,] RandomFlipHorizontal(float[,,,] images, float probability = 0.5f)
    {
        // Flip images with given probability
    }

    public static float[,,,] RandomRotate(float[,,,] images, float maxAngle = 15f)
    {
        // Rotate images by random angle
    }

    // For text data
    public static int[] RandomTokenDrop(int[] tokens, float probability = 0.1f)
    {
        // Drop tokens with given probability
    }

    // For audio data
    public static float[] AddNoise(float[] audio, float snr = 20f)
    {
        // Add Gaussian noise
    }
}
```

### Sequence Handling
```csharp
public static class SequenceUtils
{
    // Pad sequences to same length
    public static int[,] PadSequences(int[][] sequences, int maxLen, int padValue = 0)
    {
        int n = sequences.Length;
        int[,] padded = new int[n, maxLen];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < maxLen; j++)
            {
                if (j < sequences[i].Length)
                    padded[i, j] = sequences[i][j];
                else
                    padded[i, j] = padValue;
            }
        }

        return padded;
    }

    // Create sliding window sequences
    public static (int[,] X, int[,] Y) CreateSequences(int[] data, int seqLen)
    {
        int n = data.Length - seqLen;
        int[,] X = new int[n, seqLen];
        int[,] Y = new int[n, 1];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < seqLen; j++)
                X[i, j] = data[i + j];
            Y[i, 0] = data[i + seqLen];
        }

        return (X, Y);
    }
}
```

## File Format Loaders

### CSV Loader
```csharp
public static (float[,] data, float[,] labels) LoadCSV(
    string path,
    int[] featureCols,
    int[] labelCols,
    bool hasHeader = true)
{
    var lines = File.ReadAllLines(path);
    int start = hasHeader ? 1 : 0;
    int n = lines.Length - start;

    float[,] data = new float[n, featureCols.Length];
    float[,] labels = new float[n, labelCols.Length];

    for (int i = start; i < lines.Length; i++)
    {
        var values = lines[i].Split(',');
        int row = i - start;

        for (int j = 0; j < featureCols.Length; j++)
            data[row, j] = float.Parse(values[featureCols[j]]);

        for (int j = 0; j < labelCols.Length; j++)
            labels[row, j] = float.Parse(values[labelCols[j]]);
    }

    return (data, labels);
}
```

## Training Loop Integration

```csharp
// Example training loop with dataset
var dataset = new CustomDataset("data/train.csv");
dataset.BatchSize = 32;

for (int epoch = 0; epoch < numEpochs; epoch++)
{
    dataset.Shuffle();
    float epochLoss = 0f;

    foreach (var (X, Y) in dataset.GetBatches())
    {
        float loss = model.TrainOnBatch(X, Y);
        epochLoss += loss;
    }

    Console.WriteLine($"Epoch {epoch}: Loss = {epochLoss / dataset.NumBatches:F4}");
}
```

## Related Agents

- Use `/add-modality` for modality-specific preprocessing
- Use `/benchmark` to measure data loading throughput
- Use `/debug-training` if data issues cause training problems

## Quality Checklist

- [ ] Dataset implements required interface
- [ ] Shuffling works correctly
- [ ] Batching handles last incomplete batch
- [ ] Memory efficient for large datasets
- [ ] Preprocessing is reproducible
- [ ] Statistics saved for inference normalization
- [ ] Works with existing training pipeline
