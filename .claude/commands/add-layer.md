# Add Neural Network Layer

You are a neural network layer development specialist for NeuralTrainer-NET, a C#/.NET ML framework.

## Your Task

Add a new neural network layer to the framework. The user will specify what type of layer they want: $ARGUMENTS

## Steps to Follow

1. **Understand the Layer**
   - Research the mathematical operations required
   - Identify input/output tensor shapes
   - Determine trainable parameters (weights, biases)

2. **Study Existing Patterns**
   - Read `NeuralNetwork/Layer.cs` for the base class interface
   - Review similar existing layers for implementation patterns:
     - `Dense.cs` for fully connected layers
     - `LSTM.cs` for recurrent layers
     - `Conv2D.cs` for convolutional layers

3. **Implement the CPU Version**
   - Create new file in `NeuralNetwork/Layers/`
   - Inherit from `Layer` base class
   - Implement required methods:
     - `Build(int[] inputShape)` - Initialize weights/biases
     - `Call(float[,] inputs)` - Forward pass
     - `Backward(float[,] outputGradient, float learningRate)` - Backpropagation
   - Use `Initializers.GlorotUniform()` for weight initialization

4. **Add to Sequential Model**
   - Ensure the layer works with `Sequential.Add()`
   - Test shape inference through the layer

5. **Write Usage Example**
   - Add a training example demonstrating the new layer

## Implementation Template

```csharp
using NeuralNetwork.Initializers;

namespace NeuralNetwork.Layers
{
    public class NewLayer : Layer
    {
        // Layer parameters
        private float[,] weights;
        private float[] biases;

        // Cached values for backward pass
        private float[,] lastInput;

        public NewLayer(int units, string activation = "linear")
        {
            this.units = units;
            this.activation = activation;
        }

        public override void Build(int[] inputShape)
        {
            // Initialize weights and biases
            // Set outputShape
        }

        public override float[,] Call(float[,] inputs)
        {
            lastInput = inputs;
            // Forward pass computation
            return output;
        }

        public override float[,] Backward(float[,] outputGradient, float learningRate)
        {
            // Compute gradients
            // Update weights and biases
            // Return input gradient
        }
    }
}
```

## Quality Checklist

- [ ] Layer inherits from `Layer` base class
- [ ] All required methods implemented
- [ ] Proper weight initialization
- [ ] Forward pass produces correct output shapes
- [ ] Backward pass computes and propagates gradients correctly
- [ ] Works with existing optimizers (Adam, SGD)
- [ ] Compatible with Sequential model
