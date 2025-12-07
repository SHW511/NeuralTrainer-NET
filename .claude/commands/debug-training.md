# Debug Training

You are a machine learning debugging specialist for NeuralTrainer-NET, helping to diagnose and fix training issues.

## Your Task

Debug training issues in neural network models. The user will describe their problem: $ARGUMENTS

## Common Training Issues

| Symptom | Likely Causes | Solutions |
|---------|--------------|-----------|
| Loss is NaN | Exploding gradients, div by zero | Gradient clipping, lower LR, check data |
| Loss stuck high | Learning rate too low, vanishing gradients | Higher LR, skip connections, init fix |
| Loss oscillates | Learning rate too high | Lower LR, use scheduler |
| Overfitting | Model too large, not enough data | Regularization, dropout, augmentation |
| Slow convergence | Poor initialization, LR too low | Better init, LR warmup, batch norm |

## Diagnostic Steps

1. **Gather Information**
   - What model architecture?
   - What loss function and optimizer?
   - What are the hyperparameters?
   - What does the training curve look like?
   - What is the data like?

2. **Check Common Issues**
   - Data preprocessing correct?
   - Weight initialization appropriate?
   - Learning rate in reasonable range?
   - Gradient flow unobstructed?

3. **Add Diagnostics**
   - Log gradient norms
   - Monitor weight statistics
   - Track layer activations
   - Check for dead neurons

4. **Fix and Verify**
   - Apply targeted fixes
   - Compare before/after
   - Document solution

## Diagnostic Code

### Gradient Monitoring
```csharp
public static class GradientMonitor
{
    public static void LogGradientNorms(Sequential model)
    {
        Console.WriteLine("=== Gradient Norms ===");
        for (int i = 0; i < model.Layers.Count; i++)
        {
            var layer = model.Layers[i];
            if (layer.WeightGradients != null)
            {
                float norm = ComputeNorm(layer.WeightGradients);
                Console.WriteLine($"Layer {i} ({layer.GetType().Name}): {norm:E4}");

                if (float.IsNaN(norm))
                    Console.WriteLine("  WARNING: NaN gradient!");
                else if (norm > 100)
                    Console.WriteLine("  WARNING: Exploding gradient!");
                else if (norm < 1e-7)
                    Console.WriteLine("  WARNING: Vanishing gradient!");
            }
        }
    }

    private static float ComputeNorm(float[,] arr)
    {
        float sum = 0;
        for (int i = 0; i < arr.GetLength(0); i++)
            for (int j = 0; j < arr.GetLength(1); j++)
                sum += arr[i, j] * arr[i, j];
        return MathF.Sqrt(sum);
    }
}
```

### Weight Statistics
```csharp
public static class WeightMonitor
{
    public static void LogWeightStats(Sequential model)
    {
        Console.WriteLine("=== Weight Statistics ===");
        for (int i = 0; i < model.Layers.Count; i++)
        {
            var layer = model.Layers[i];
            if (layer.Weights != null)
            {
                var (min, max, mean, std) = ComputeStats(layer.Weights);
                Console.WriteLine($"Layer {i}: min={min:F4}, max={max:F4}, mean={mean:F4}, std={std:F4}");

                if (std < 1e-6)
                    Console.WriteLine("  WARNING: Weights nearly constant!");
                if (max > 100 || min < -100)
                    Console.WriteLine("  WARNING: Very large weights!");
            }
        }
    }

    private static (float min, float max, float mean, float std) ComputeStats(float[,] arr)
    {
        float min = float.MaxValue, max = float.MinValue, sum = 0;
        int count = arr.GetLength(0) * arr.GetLength(1);

        foreach (float val in arr)
        {
            min = Math.Min(min, val);
            max = Math.Max(max, val);
            sum += val;
        }

        float mean = sum / count;
        float variance = 0;
        foreach (float val in arr)
            variance += (val - mean) * (val - mean);

        return (min, max, mean, MathF.Sqrt(variance / count));
    }
}
```

### Activation Monitoring
```csharp
public static class ActivationMonitor
{
    public static void LogActivationStats(float[,] activations, string layerName)
    {
        float zeros = 0, total = activations.Length;
        float min = float.MaxValue, max = float.MinValue;

        foreach (float val in activations)
        {
            if (val == 0) zeros++;
            min = Math.Min(min, val);
            max = Math.Max(max, val);
        }

        float deadRatio = zeros / total;
        Console.WriteLine($"{layerName}: range=[{min:F4}, {max:F4}], dead neurons={deadRatio:P1}");

        if (deadRatio > 0.5)
            Console.WriteLine("  WARNING: Many dead neurons (>50%)!");
    }
}
```

### NaN/Inf Checker
```csharp
public static class NumericalChecker
{
    public static bool CheckForNaN(float[,] arr, string name)
    {
        foreach (float val in arr)
        {
            if (float.IsNaN(val))
            {
                Console.WriteLine($"NaN found in {name}!");
                return true;
            }
            if (float.IsInfinity(val))
            {
                Console.WriteLine($"Infinity found in {name}!");
                return true;
            }
        }
        return false;
    }
}
```

## Common Fixes

### Gradient Clipping
```csharp
public static void ClipGradients(float[,] gradients, float maxNorm)
{
    float norm = ComputeNorm(gradients);
    if (norm > maxNorm)
    {
        float scale = maxNorm / norm;
        for (int i = 0; i < gradients.GetLength(0); i++)
            for (int j = 0; j < gradients.GetLength(1); j++)
                gradients[i, j] *= scale;
    }
}
```

### Learning Rate Warmup
```csharp
public static float WarmupLR(int step, int warmupSteps, float targetLR)
{
    if (step < warmupSteps)
        return targetLR * (step + 1) / warmupSteps;
    return targetLR;
}
```

### He Initialization (for ReLU)
```csharp
public static float[,] HeInitialization(int inputSize, int outputSize)
{
    float stddev = MathF.Sqrt(2.0f / inputSize);
    var rng = new Random();
    var weights = new float[inputSize, outputSize];

    for (int i = 0; i < inputSize; i++)
        for (int j = 0; j < outputSize; j++)
            weights[i, j] = (float)(rng.NextDouble() * 2 - 1) * stddev;

    return weights;
}
```

## Debugging Checklist

### Data Issues
- [ ] Check for NaN/Inf in input data
- [ ] Verify normalization (mean ≈ 0, std ≈ 1)
- [ ] Check label encoding is correct
- [ ] Ensure batch sampling is random

### Model Issues
- [ ] Weight initialization appropriate for activation
- [ ] No divide-by-zero in forward pass
- [ ] Shapes propagate correctly
- [ ] CUDA vs CPU produce same results

### Training Issues
- [ ] Learning rate in reasonable range (1e-4 to 1e-2)
- [ ] Loss function matches task
- [ ] Optimizer state initialized
- [ ] Gradients flowing to all layers

### Numerical Issues
- [ ] Epsilon added where needed
- [ ] Log applied to clamped values
- [ ] Softmax with log-sum-exp trick
- [ ] Float32 precision sufficient

## Related Agents

- Use `/test-model` to validate layer implementations
- Use `/benchmark` to check if issue is performance-related
- Use `/architecture` to review model design
- Use `/add-layer` if layer implementation is buggy

## Output Format

When debugging, report:

1. **Issue Identified**: What the problem is
2. **Root Cause**: Why it's happening
3. **Evidence**: Diagnostic outputs that confirm
4. **Solution**: Code changes to fix
5. **Verification**: How to confirm fix works
