using System.Collections.Generic;

namespace NeuralNetwork
{
    public abstract class Layer
    {
        public bool Built { get; protected set; } = false;

        public int InputDim { get; protected set; }

        public int OutputDim { get; protected set; }

        public int[] InputShape { get; protected set; }
        public float[,] Weights { get; protected set; }

        public abstract void Build(int[] inputShape);
        public abstract float[,] Call(float[,] inputs);
        public abstract int[] GetOutputShape(int[] inputShape);
        public virtual Dictionary<string, object> GetConfig()
        {
            return new Dictionary<string, object>();
        }

        public abstract float[,] Backward(float[,] gradient);
    }
}
