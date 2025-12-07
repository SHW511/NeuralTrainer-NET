# Benchmark

You are a performance benchmarking specialist for NeuralTrainer-NET, helping to measure and optimize performance.

## Your Task

Run benchmarks and performance analysis. The user will specify what to benchmark: $ARGUMENTS

## Benchmark Categories

1. **Layer Performance**: Forward/backward pass timing
2. **Training Throughput**: Samples/second, batches/second
3. **Memory Usage**: GPU and CPU memory consumption
4. **CPU vs CUDA**: Speedup comparison
5. **Scaling**: Performance vs batch size, model size

## Steps to Follow

1. **Identify Benchmark Target**
   - Which component to benchmark
   - What metrics to measure
   - What configurations to test

2. **Setup Benchmarks**
   - Create controlled test environment
   - Warm up GPU/JIT
   - Define input sizes

3. **Run Measurements**
   - Execute multiple iterations
   - Collect timing data
   - Record memory usage

4. **Analyze and Report**
   - Calculate statistics (mean, std, percentiles)
   - Compare configurations
   - Identify bottlenecks

## Benchmark Framework

```csharp
using System.Diagnostics;

namespace NeuralNetwork.Benchmarks
{
    public class Benchmark
    {
        private Stopwatch sw = new Stopwatch();
        private List<double> measurements = new List<double>();

        public int WarmupIterations { get; set; } = 10;
        public int MeasureIterations { get; set; } = 100;

        public void Run(string name, Action action)
        {
            Console.WriteLine($"=== Benchmark: {name} ===");

            // Warmup
            Console.Write($"Warming up ({WarmupIterations} iterations)... ");
            for (int i = 0; i < WarmupIterations; i++)
                action();
            Console.WriteLine("done");

            // Measure
            measurements.Clear();
            for (int i = 0; i < MeasureIterations; i++)
            {
                sw.Restart();
                action();
                sw.Stop();
                measurements.Add(sw.Elapsed.TotalMilliseconds);
            }

            // Report
            ReportResults();
        }

        public void Run<T>(string name, Func<T> func)
        {
            Run(name, () => { var _ = func(); });
        }

        private void ReportResults()
        {
            measurements.Sort();
            double mean = measurements.Average();
            double std = Math.Sqrt(measurements.Average(x => (x - mean) * (x - mean)));
            double p50 = measurements[measurements.Count / 2];
            double p95 = measurements[(int)(measurements.Count * 0.95)];
            double p99 = measurements[(int)(measurements.Count * 0.99)];

            Console.WriteLine($"  Mean:    {mean:F3} ms");
            Console.WriteLine($"  Std:     {std:F3} ms");
            Console.WriteLine($"  P50:     {p50:F3} ms");
            Console.WriteLine($"  P95:     {p95:F3} ms");
            Console.WriteLine($"  P99:     {p99:F3} ms");
            Console.WriteLine($"  Throughput: {1000.0 / mean:F1} ops/sec");
        }
    }
}
```

## Layer Benchmarks

```csharp
public static void BenchmarkLayers()
{
    var benchmark = new Benchmark { MeasureIterations = 500 };
    int batchSize = 32;

    // Dense layer
    var dense = new Dense(256, activation: "relu");
    dense.Build(new int[] { 128 });
    float[,] denseInput = CreateRandomArray(batchSize, 128);

    benchmark.Run("Dense Forward (128->256)", () => dense.Call(denseInput));

    float[,] denseGrad = CreateRandomArray(batchSize, 256);
    benchmark.Run("Dense Backward", () => dense.Backward(denseGrad, 0.001f));

    // LSTM layer
    var lstm = new LSTM(128);
    lstm.Build(new int[] { 50, 64 }); // seq_len=50, features=64
    float[,,] lstmInput = CreateRandomArray3D(batchSize, 50, 64);

    benchmark.Run("LSTM Forward (50x64->128)", () => lstm.Call(lstmInput));

    // Dense CUDA
    var denseCuda = new DenseCuda(256, activation: "relu");
    denseCuda.Build(new int[] { 128 });

    benchmark.Run("DenseCuda Forward (128->256)", () => denseCuda.Call(denseInput));

    // Compare CPU vs CUDA
    Console.WriteLine("\n=== CPU vs CUDA Comparison ===");
    // ... comparison code
}
```

## Training Throughput Benchmark

```csharp
public static void BenchmarkTraining()
{
    var benchmark = new Benchmark { MeasureIterations = 100 };

    // Create model
    var model = new Sequential();
    model.Add(new Dense(256, activation: "relu"));
    model.Add(new Dense(128, activation: "relu"));
    model.Add(new Dense(10, activation: "softmax"));
    model.Build(new int[] { 784 });
    model.Compile(new CategoricalCrossentropy(), new Adam(0.001f));

    // Test different batch sizes
    int[] batchSizes = { 16, 32, 64, 128, 256 };

    foreach (int batchSize in batchSizes)
    {
        float[,] X = CreateRandomArray(batchSize, 784);
        float[,] Y = CreateOneHot(batchSize, 10);

        benchmark.Run($"Training batch_size={batchSize}", () => model.TrainOnBatch(X, Y));

        double samplesPerSec = batchSize * 1000.0 / benchmark.MeanMs;
        Console.WriteLine($"  Samples/sec: {samplesPerSec:F0}");
    }
}
```

## Memory Profiling

```csharp
public static void ProfileMemory()
{
    Console.WriteLine("=== Memory Profile ===");

    // Force GC and baseline
    GC.Collect();
    GC.WaitForPendingFinalizers();
    long baseline = GC.GetTotalMemory(true);
    Console.WriteLine($"Baseline: {baseline / 1024.0 / 1024.0:F2} MB");

    // Create model
    var model = new Sequential();
    model.Add(new Embedding(10000, 256));
    model.Add(new LSTM(512));
    model.Add(new Dense(10000, activation: "softmax"));
    model.Build(new int[] { 100 }); // seq_len=100

    GC.Collect();
    long afterModel = GC.GetTotalMemory(true);
    Console.WriteLine($"After model: {afterModel / 1024.0 / 1024.0:F2} MB");
    Console.WriteLine($"Model size: {(afterModel - baseline) / 1024.0 / 1024.0:F2} MB");

    // Forward pass with batch
    float[,] input = new float[32, 100];
    var output = model.Forward(input);

    GC.Collect();
    long afterForward = GC.GetTotalMemory(true);
    Console.WriteLine($"After forward: {afterForward / 1024.0 / 1024.0:F2} MB");
    Console.WriteLine($"Activations: {(afterForward - afterModel) / 1024.0 / 1024.0:F2} MB");
}
```

## CUDA Memory Profiling

```csharp
public static void ProfileCudaMemory()
{
    var context = new CudaContext(0);

    Console.WriteLine("=== CUDA Memory Profile ===");
    Console.WriteLine($"Device: {context.GetDeviceInfo().DeviceName}");
    Console.WriteLine($"Total memory: {context.GetDeviceInfo().TotalGlobalMemory / 1024.0 / 1024.0:F0} MB");

    // Get free memory before
    context.GetDeviceMemoryInfo(out long freeBefore, out long total);
    Console.WriteLine($"Free before: {freeBefore / 1024.0 / 1024.0:F0} MB");

    // Allocate model on GPU
    var model = new TransformerLMCuda(config);

    context.GetDeviceMemoryInfo(out long freeAfter, out long _);
    Console.WriteLine($"Free after model: {freeAfter / 1024.0 / 1024.0:F0} MB");
    Console.WriteLine($"Model GPU memory: {(freeBefore - freeAfter) / 1024.0 / 1024.0:F0} MB");
}
```

## Scaling Analysis

```csharp
public static void AnalyzeScaling()
{
    Console.WriteLine("=== Scaling Analysis ===");

    var benchmark = new Benchmark { MeasureIterations = 50 };

    // Model size scaling
    Console.WriteLine("\n--- Hidden Size Scaling ---");
    int[] hiddenSizes = { 64, 128, 256, 512, 1024 };
    foreach (int size in hiddenSizes)
    {
        var layer = new Dense(size);
        layer.Build(new int[] { size });
        float[,] input = CreateRandomArray(32, size);

        benchmark.Run($"Dense {size}x{size}", () => layer.Call(input));
    }

    // Batch size scaling
    Console.WriteLine("\n--- Batch Size Scaling ---");
    int[] batchSizes = { 1, 8, 16, 32, 64, 128 };
    var fixedLayer = new Dense(256);
    fixedLayer.Build(new int[] { 256 });

    foreach (int bs in batchSizes)
    {
        float[,] input = CreateRandomArray(bs, 256);
        benchmark.Run($"Batch size {bs}", () => fixedLayer.Call(input));
        Console.WriteLine($"  Per-sample: {benchmark.MeanMs / bs:F4} ms");
    }
}
```

## Output Format

```
=== Benchmark Results ===

Layer Performance:
| Layer      | Forward (ms) | Backward (ms) | CUDA Speedup |
|------------|--------------|---------------|--------------|
| Dense 256  | 0.45         | 0.82          | 5.2x         |
| LSTM 512   | 2.34         | 4.56          | 8.1x         |
| Attention  | 1.23         | 2.45          | 6.7x         |

Training Throughput:
| Batch Size | Samples/sec | GPU Util |
|------------|-------------|----------|
| 32         | 1,234       | 45%      |
| 64         | 2,345       | 72%      |
| 128        | 4,123       | 91%      |

Memory Usage:
| Component     | CPU (MB) | GPU (MB) |
|---------------|----------|----------|
| Model weights | 45.2     | 45.2     |
| Activations   | 128.4    | 128.4    |
| Gradients     | 45.2     | 45.2     |
| Total         | 218.8    | 218.8    |
```

## Related Agents

- Use `/add-cuda` to GPU-accelerate bottlenecks
- Use `/debug-training` if performance issues affect convergence
- Use `/architecture` to design more efficient models

## Quality Checklist

- [ ] Warmup iterations included
- [ ] Multiple measurements for statistics
- [ ] Both CPU and CUDA timed
- [ ] Memory usage tracked
- [ ] Results clearly formatted
- [ ] Bottlenecks identified
- [ ] Recommendations provided
