# Add Optimizer

You are an optimization algorithm specialist for NeuralTrainer-NET, helping to implement new training optimizers.

## Your Task

Add a new optimizer to the framework. The user will specify which optimizer: $ARGUMENTS

## Common Optimizers to Implement

| Optimizer | Key Features | Use Case |
|-----------|--------------|----------|
| AdaGrad | Per-parameter learning rates | Sparse features |
| RMSprop | Moving average of squared gradients | RNN training |
| AdamW | Adam with decoupled weight decay | Transformers |
| Nadam | Adam + Nesterov momentum | General purpose |
| LAMB | Layer-wise Adaptive Rate Scaling | Large batch training |
| RAdam | Rectified Adam | Stable early training |
| Lookahead | Wrapper for any optimizer | Better generalization |

## Steps to Follow

1. **Research the Optimizer**
   - Understand the mathematical formulation
   - Identify hyperparameters (learning rate, momentum, decay, etc.)
   - Study convergence properties
   - Note numerical stability considerations

2. **Study Existing Patterns**
   - Read `NeuralNetwork/Optimizers/Optimizer.cs` for base class
   - Review `Adam.cs` for a complex optimizer example
   - Review `SGD.cs` for a simple optimizer example

3. **Implement the Optimizer**
   - Create new file in `NeuralNetwork/Optimizers/`
   - Inherit from `Optimizer` base class
   - Implement `Update()` method for weight updates
   - Store optimizer state (momentum buffers, etc.)

4. **Test Convergence**
   - Use `/test-model` to validate on simple problems
   - Compare convergence to existing optimizers
   - Verify numerical stability

## Implementation Template

```csharp
namespace NeuralNetwork.Optimizers
{
    public class NewOptimizer : Optimizer
    {
        private float learningRate;
        private float beta1;  // First moment decay
        private float beta2;  // Second moment decay
        private float epsilon;  // Numerical stability

        // Per-parameter state
        private Dictionary<int, float[,]> m;  // First moment
        private Dictionary<int, float[,]> v;  // Second moment
        private int t;  // Timestep

        public NewOptimizer(float learningRate = 0.001f,
                           float beta1 = 0.9f,
                           float beta2 = 0.999f,
                           float epsilon = 1e-8f)
        {
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
            this.m = new Dictionary<int, float[,]>();
            this.v = new Dictionary<int, float[,]>();
            this.t = 0;
        }

        public override void Update(int paramId, float[,] weights, float[,] gradients)
        {
            // Initialize state if first update
            if (!m.ContainsKey(paramId))
            {
                m[paramId] = new float[weights.GetLength(0), weights.GetLength(1)];
                v[paramId] = new float[weights.GetLength(0), weights.GetLength(1)];
            }

            t++;

            // Update first moment estimate
            // Update second moment estimate
            // Compute bias-corrected estimates
            // Apply weight update

            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    // m[paramId][i, j] = beta1 * m[paramId][i, j] + (1 - beta1) * gradients[i, j];
                    // v[paramId][i, j] = beta2 * v[paramId][i, j] + (1 - beta2) * gradients[i, j] * gradients[i, j];
                    // float mHat = m[paramId][i, j] / (1 - MathF.Pow(beta1, t));
                    // float vHat = v[paramId][i, j] / (1 - MathF.Pow(beta2, t));
                    // weights[i, j] -= learningRate * mHat / (MathF.Sqrt(vHat) + epsilon);
                }
            }
        }

        // For 1D parameters (biases)
        public override void Update(int paramId, float[] weights, float[] gradients)
        {
            // Similar implementation for 1D arrays
        }

        public override void Reset()
        {
            m.Clear();
            v.Clear();
            t = 0;
        }
    }
}
```

## AdamW Implementation (Weight Decay)

```csharp
public class AdamW : Optimizer
{
    private float weightDecay;

    public override void Update(int paramId, float[,] weights, float[,] gradients)
    {
        // Standard Adam update...

        // Decoupled weight decay (applied directly to weights)
        for (int i = 0; i < weights.GetLength(0); i++)
        {
            for (int j = 0; j < weights.GetLength(1); j++)
            {
                weights[i, j] -= learningRate * weightDecay * weights[i, j];
            }
        }
    }
}
```

## Learning Rate Scheduling Integration

Optimizers should work with `LearningRateScheduler`:

```csharp
// In Sequential.Compile or training loop
var scheduler = new LearningRateScheduler(
    initialLR: 0.001f,
    schedule: LearningRateScheduler.CosineAnnealing(totalSteps: 10000)
);

// During training
float currentLR = scheduler.GetLR(currentStep);
optimizer.SetLearningRate(currentLR);
```

## Related Agents

- Use `/test-model` to validate optimizer convergence
- Use `/benchmark` to compare optimizer performance
- Use `/debug-training` if training doesn't converge

## Quality Checklist

- [ ] Inherits from `Optimizer` base class
- [ ] Proper hyperparameter initialization
- [ ] State management for momentum/adaptive rates
- [ ] Handles both 2D weights and 1D biases
- [ ] Numerical stability (epsilon, gradient clipping)
- [ ] Reset method clears all state
- [ ] Tested on XOR and other simple problems
- [ ] Convergence comparable to existing optimizers
