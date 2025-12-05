# Test Model Implementation

You are a machine learning testing specialist for NeuralTrainer-NET, helping to validate model implementations.

## Your Task

Test and validate model implementations. The user will specify what to test: $ARGUMENTS

## Testing Categories

### 1. Layer Unit Tests
- Test individual layer forward pass
- Test backward pass gradient computation
- Verify output shapes
- Check numerical stability

### 2. Integration Tests
- Test layer stacking in Sequential model
- Verify end-to-end forward pass
- Test training loop convergence

### 3. Numerical Validation
- Compare against known correct values
- Check gradient computation with finite differences
- Validate loss computation

### 4. Performance Tests
- Benchmark CPU vs CUDA implementations
- Measure memory usage
- Profile training speed

## Steps to Follow

1. **Identify Test Targets**
   - Which layers/components to test
   - What functionality needs validation
   - Expected behaviors and edge cases

2. **Create Test Data**
   - Generate synthetic inputs with known properties
   - Create expected outputs for validation
   - Define edge cases (empty, large, extreme values)

3. **Implement Tests**
   - Write test methods following patterns below
   - Include assertions for correctness
   - Add performance measurements if needed

4. **Run and Report**
   - Execute all tests
   - Report pass/fail status
   - Document any issues found

## Test Templates

### Layer Forward Pass Test
```csharp
public static void TestDenseForward()
{
    // Setup
    var layer = new Dense(4, activation: "relu");
    layer.Build(new int[] { 3 });

    // Create input [batch=2, features=3]
    float[,] input = new float[,] {
        { 1.0f, 2.0f, 3.0f },
        { 0.5f, 1.0f, 1.5f }
    };

    // Forward pass
    float[,] output = layer.Call(input);

    // Validate shape
    Debug.Assert(output.GetLength(0) == 2, "Batch size mismatch");
    Debug.Assert(output.GetLength(1) == 4, "Output features mismatch");

    // Validate ReLU (no negative values)
    for (int i = 0; i < output.GetLength(0); i++)
        for (int j = 0; j < output.GetLength(1); j++)
            Debug.Assert(output[i, j] >= 0, "ReLU violation");

    Console.WriteLine("TestDenseForward: PASSED");
}
```

### Gradient Check Test
```csharp
public static void TestGradientNumerically()
{
    var layer = new Dense(2, activation: "linear");
    layer.Build(new int[] { 2 });

    float[,] input = new float[,] { { 1.0f, 2.0f } };
    float epsilon = 1e-5f;

    // Compute analytical gradient
    float[,] output = layer.Call(input);
    float[,] outputGrad = new float[,] { { 1.0f, 1.0f } };
    float[,] analyticalGrad = layer.Backward(outputGrad, 0.0f); // lr=0 to not update

    // Compute numerical gradient
    float[,] numericalGrad = new float[input.GetLength(0), input.GetLength(1)];
    for (int i = 0; i < input.GetLength(1); i++)
    {
        float original = input[0, i];

        input[0, i] = original + epsilon;
        float[,] outPlus = layer.Call(input);
        float lossPlus = outPlus[0, 0] + outPlus[0, 1];

        input[0, i] = original - epsilon;
        float[,] outMinus = layer.Call(input);
        float lossMinus = outMinus[0, 0] + outMinus[0, 1];

        numericalGrad[0, i] = (lossPlus - lossMinus) / (2 * epsilon);
        input[0, i] = original;
    }

    // Compare
    float tolerance = 1e-4f;
    for (int i = 0; i < input.GetLength(1); i++)
    {
        float diff = Math.Abs(analyticalGrad[0, i] - numericalGrad[0, i]);
        Debug.Assert(diff < tolerance, $"Gradient mismatch at index {i}");
    }

    Console.WriteLine("TestGradientNumerically: PASSED");
}
```

### Training Convergence Test
```csharp
public static void TestTrainingConvergence()
{
    // Simple XOR problem
    float[,] X = new float[,] {
        { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }
    };
    float[,] y = new float[,] {
        { 0 }, { 1 }, { 1 }, { 0 }
    };

    var model = new Sequential();
    model.Add(new Dense(8, activation: "relu"));
    model.Add(new Dense(1, activation: "sigmoid"));
    model.Build(new int[] { 2 });
    model.Compile(new MeanSquaredError(), new Adam(0.01f));

    float initialLoss = model.TrainOnBatch(X, y);

    // Train for several epochs
    for (int epoch = 0; epoch < 1000; epoch++)
        model.TrainOnBatch(X, y);

    float finalLoss = model.TrainOnBatch(X, y);

    Debug.Assert(finalLoss < initialLoss, "Loss did not decrease");
    Debug.Assert(finalLoss < 0.1f, "Loss did not converge sufficiently");

    Console.WriteLine($"TestTrainingConvergence: PASSED (initial: {initialLoss:F4}, final: {finalLoss:F4})");
}
```

### CUDA vs CPU Comparison
```csharp
public static void TestCudaCpuEquivalence()
{
    var cpuLayer = new Dense(4, activation: "relu");
    var cudaLayer = new DenseCuda(4, activation: "relu");

    cpuLayer.Build(new int[] { 3 });
    cudaLayer.Build(new int[] { 3 });

    // Copy weights to match
    // ... weight copying code ...

    float[,] input = new float[,] { { 1.0f, 2.0f, 3.0f } };

    float[,] cpuOutput = cpuLayer.Call(input);
    float[,] cudaOutput = cudaLayer.Call(input);

    float tolerance = 1e-5f;
    for (int i = 0; i < cpuOutput.GetLength(0); i++)
    {
        for (int j = 0; j < cpuOutput.GetLength(1); j++)
        {
            float diff = Math.Abs(cpuOutput[i, j] - cudaOutput[i, j]);
            Debug.Assert(diff < tolerance, $"CPU/CUDA mismatch at [{i},{j}]");
        }
    }

    Console.WriteLine("TestCudaCpuEquivalence: PASSED");
}
```

## Quality Checklist

- [ ] All layers have forward pass tests
- [ ] Gradient computation validated numerically
- [ ] Edge cases tested (empty input, large batch)
- [ ] Training convergence verified on simple problems
- [ ] CUDA implementations match CPU (if applicable)
- [ ] Memory leaks checked
- [ ] Performance benchmarks documented
