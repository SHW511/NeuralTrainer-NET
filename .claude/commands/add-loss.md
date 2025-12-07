# Add Loss Function

You are a loss function specialist for NeuralTrainer-NET, helping to implement new loss functions for training.

## Your Task

Add a new loss function to the framework. The user will specify which loss: $ARGUMENTS

## Common Loss Functions

| Loss Function | Use Case | Output |
|---------------|----------|--------|
| BinaryCrossentropy | Binary classification | Sigmoid |
| SparseCategoricalCrossentropy | Multi-class (integer labels) | Softmax |
| Huber | Regression (robust to outliers) | Linear |
| MAE (L1) | Regression | Linear |
| Focal | Imbalanced classification | Softmax |
| CTC | Sequence-to-sequence (ASR) | Softmax |
| Triplet | Embedding learning | N/A |
| Contrastive | Siamese networks | N/A |
| KL Divergence | Distribution matching | Softmax |
| Hinge | SVM-style classification | Linear |

## Steps to Follow

1. **Research the Loss Function**
   - Understand the mathematical formulation
   - Know the gradient computation (backward pass)
   - Identify numerical stability issues
   - Understand when to use this loss

2. **Study Existing Patterns**
   - Read `NeuralNetwork/Losses/Loss.cs` for base class
   - Review `MeanSquaredError.cs` for regression loss
   - Review `CategoricalCrossentropy.cs` for classification loss

3. **Implement the Loss**
   - Create new file in `NeuralNetwork/Losses/`
   - Inherit from `Loss` base class
   - Implement `Compute()` for forward pass
   - Implement `Gradient()` for backward pass

4. **Test Correctness**
   - Verify gradient numerically
   - Test edge cases (zeros, very small values)
   - Ensure training converges

## Implementation Template

```csharp
namespace NeuralNetwork.Losses
{
    public class NewLoss : Loss
    {
        private float epsilon = 1e-7f;  // Numerical stability

        // Optional hyperparameters
        private float alpha;
        private float gamma;

        public NewLoss(float alpha = 1.0f, float gamma = 2.0f)
        {
            this.alpha = alpha;
            this.gamma = gamma;
        }

        public override float Compute(float[,] predictions, float[,] targets)
        {
            int batchSize = predictions.GetLength(0);
            int outputSize = predictions.GetLength(1);
            float totalLoss = 0.0f;

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    // Compute loss for each element
                    // totalLoss += ...
                }
            }

            return totalLoss / batchSize;
        }

        public override float[,] Gradient(float[,] predictions, float[,] targets)
        {
            int batchSize = predictions.GetLength(0);
            int outputSize = predictions.GetLength(1);
            float[,] gradients = new float[batchSize, outputSize];

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    // Compute gradient for each element
                    // gradients[i, j] = ...
                }
            }

            // Average over batch
            for (int i = 0; i < batchSize; i++)
                for (int j = 0; j < outputSize; j++)
                    gradients[i, j] /= batchSize;

            return gradients;
        }
    }
}
```

## Specific Loss Implementations

### Binary Cross-Entropy
```csharp
public class BinaryCrossentropy : Loss
{
    public override float Compute(float[,] predictions, float[,] targets)
    {
        float loss = 0.0f;
        int n = predictions.GetLength(0);

        for (int i = 0; i < n; i++)
        {
            float p = Math.Clamp(predictions[i, 0], epsilon, 1 - epsilon);
            float t = targets[i, 0];
            loss -= t * MathF.Log(p) + (1 - t) * MathF.Log(1 - p);
        }

        return loss / n;
    }

    public override float[,] Gradient(float[,] predictions, float[,] targets)
    {
        float[,] grad = new float[predictions.GetLength(0), 1];
        int n = predictions.GetLength(0);

        for (int i = 0; i < n; i++)
        {
            float p = Math.Clamp(predictions[i, 0], epsilon, 1 - epsilon);
            float t = targets[i, 0];
            grad[i, 0] = (-t / p + (1 - t) / (1 - p)) / n;
        }

        return grad;
    }
}
```

### Focal Loss (for imbalanced data)
```csharp
public class FocalLoss : Loss
{
    private float alpha;  // Class weight
    private float gamma;  // Focusing parameter

    public FocalLoss(float alpha = 0.25f, float gamma = 2.0f)
    {
        this.alpha = alpha;
        this.gamma = gamma;
    }

    public override float Compute(float[,] predictions, float[,] targets)
    {
        float loss = 0.0f;
        int n = predictions.GetLength(0);

        for (int i = 0; i < n; i++)
        {
            float p = Math.Clamp(predictions[i, 0], epsilon, 1 - epsilon);
            float t = targets[i, 0];
            float pt = t * p + (1 - t) * (1 - p);
            float alphaT = t * alpha + (1 - t) * (1 - alpha);
            loss -= alphaT * MathF.Pow(1 - pt, gamma) * MathF.Log(pt);
        }

        return loss / n;
    }
}
```

### Huber Loss (robust regression)
```csharp
public class HuberLoss : Loss
{
    private float delta;

    public HuberLoss(float delta = 1.0f)
    {
        this.delta = delta;
    }

    public override float Compute(float[,] predictions, float[,] targets)
    {
        float loss = 0.0f;
        int n = predictions.GetLength(0);
        int m = predictions.GetLength(1);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                float diff = predictions[i, j] - targets[i, j];
                if (MathF.Abs(diff) <= delta)
                    loss += 0.5f * diff * diff;
                else
                    loss += delta * (MathF.Abs(diff) - 0.5f * delta);
            }
        }

        return loss / n;
    }
}
```

## Numerical Stability Tips

1. **Clamp predictions**: Avoid log(0) with epsilon clamping
2. **Log-sum-exp trick**: For softmax + cross-entropy
3. **Gradient clipping**: Prevent exploding gradients
4. **Check for NaN**: Add assertions during development

## Related Agents

- Use `/test-model` to verify gradient computation
- Use `/debug-training` if loss produces NaN/Inf
- Use `/add-layer` if you need a custom output activation

## Quality Checklist

- [ ] Inherits from `Loss` base class
- [ ] Compute() returns scalar loss value
- [ ] Gradient() returns correct gradient shape
- [ ] Numerical stability (epsilon, clamping)
- [ ] Gradient verified numerically
- [ ] Edge cases handled (zeros, ones)
- [ ] Documentation with formula
- [ ] Works with existing training pipeline
