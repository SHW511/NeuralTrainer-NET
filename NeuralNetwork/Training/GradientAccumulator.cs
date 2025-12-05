using System;
using System.Collections.Generic;

namespace NeuralNetwork.Training
{
    /// <summary>
    /// Gradient accumulator for simulating larger batch sizes.
    ///
    /// When GPU memory is limited, gradient accumulation allows training with
    /// effectively larger batches by accumulating gradients over multiple
    /// forward/backward passes before updating weights.
    ///
    /// Effective batch size = micro_batch_size * accumulation_steps
    /// </summary>
    public class GradientAccumulator
    {
        private readonly int _accumulationSteps;
        private int _currentStep;

        // Accumulated gradients: name -> accumulated gradient array
        private readonly Dictionary<string, float[,]> _accumulatedGradients;

        /// <summary>
        /// Number of steps to accumulate before updating.
        /// </summary>
        public int AccumulationSteps => _accumulationSteps;

        /// <summary>
        /// Current accumulation step (0 to AccumulationSteps-1).
        /// </summary>
        public int CurrentStep => _currentStep;

        /// <summary>
        /// Whether it's time to perform a weight update.
        /// </summary>
        public bool ShouldUpdate => _currentStep >= _accumulationSteps;

        /// <summary>
        /// Create a gradient accumulator.
        /// </summary>
        /// <param name="accumulationSteps">Number of micro-batches to accumulate.</param>
        public GradientAccumulator(int accumulationSteps = 1)
        {
            if (accumulationSteps < 1)
                throw new ArgumentException("Accumulation steps must be at least 1");

            _accumulationSteps = accumulationSteps;
            _currentStep = 0;
            _accumulatedGradients = new Dictionary<string, float[,]>();
        }

        /// <summary>
        /// Accumulate gradients from a parameter set.
        /// Call this after each backward pass.
        /// </summary>
        /// <param name="parameters">List of (name, weights, gradients) tuples from the model.</param>
        public void Accumulate(List<(string name, float[,] weights, float[,] gradients)> parameters)
        {
            foreach (var (name, weights, gradients) in parameters)
            {
                if (!_accumulatedGradients.ContainsKey(name))
                {
                    // Initialize accumulated gradient array
                    _accumulatedGradients[name] = new float[gradients.GetLength(0), gradients.GetLength(1)];
                }

                var accumulated = _accumulatedGradients[name];

                // Add current gradients to accumulated
                for (int i = 0; i < gradients.GetLength(0); i++)
                {
                    for (int j = 0; j < gradients.GetLength(1); j++)
                    {
                        accumulated[i, j] += gradients[i, j];
                    }
                }
            }

            _currentStep++;
        }

        /// <summary>
        /// Get accumulated and averaged gradients, then reset.
        /// Call this when ShouldUpdate is true.
        /// </summary>
        /// <param name="parameters">Model parameters to update gradients in-place.</param>
        public void ApplyAccumulatedGradients(List<(string name, float[,] weights, float[,] gradients)> parameters)
        {
            if (_currentStep == 0)
                return;

            float scale = 1f / _currentStep;  // Average over accumulated steps

            foreach (var (name, weights, gradients) in parameters)
            {
                if (_accumulatedGradients.TryGetValue(name, out var accumulated))
                {
                    // Copy averaged accumulated gradients to the model's gradient arrays
                    for (int i = 0; i < gradients.GetLength(0); i++)
                    {
                        for (int j = 0; j < gradients.GetLength(1); j++)
                        {
                            gradients[i, j] = accumulated[i, j] * scale;
                        }
                    }
                }
            }

            Reset();
        }

        /// <summary>
        /// Reset accumulator state.
        /// </summary>
        public void Reset()
        {
            _currentStep = 0;

            // Clear accumulated gradients
            foreach (var accumulated in _accumulatedGradients.Values)
            {
                Array.Clear(accumulated, 0, accumulated.Length);
            }
        }

        /// <summary>
        /// Get the scaling factor for loss (for logging purposes).
        /// When accumulating, you may want to scale the reported loss.
        /// </summary>
        public float LossScale => 1f / _accumulationSteps;
    }

    /// <summary>
    /// Training utilities for gradient clipping.
    /// </summary>
    public static class GradientUtils
    {
        /// <summary>
        /// Clip gradients by global norm.
        /// If ||g|| > max_norm, scale all gradients by max_norm / ||g||
        /// </summary>
        /// <param name="parameters">Model parameters.</param>
        /// <param name="maxNorm">Maximum allowed gradient norm.</param>
        /// <returns>The original gradient norm (before clipping).</returns>
        public static float ClipGradientsByNorm(
            List<(string name, float[,] weights, float[,] gradients)> parameters,
            float maxNorm)
        {
            // Compute global gradient norm
            float totalNormSq = 0f;

            foreach (var (_, _, gradients) in parameters)
            {
                for (int i = 0; i < gradients.GetLength(0); i++)
                {
                    for (int j = 0; j < gradients.GetLength(1); j++)
                    {
                        totalNormSq += gradients[i, j] * gradients[i, j];
                    }
                }
            }

            float totalNorm = (float)Math.Sqrt(totalNormSq);

            // Clip if necessary
            if (totalNorm > maxNorm)
            {
                float scale = maxNorm / totalNorm;

                foreach (var (_, _, gradients) in parameters)
                {
                    for (int i = 0; i < gradients.GetLength(0); i++)
                    {
                        for (int j = 0; j < gradients.GetLength(1); j++)
                        {
                            gradients[i, j] *= scale;
                        }
                    }
                }
            }

            return totalNorm;
        }

        /// <summary>
        /// Clip gradients by value (element-wise).
        /// </summary>
        /// <param name="parameters">Model parameters.</param>
        /// <param name="clipValue">Maximum absolute gradient value.</param>
        public static void ClipGradientsByValue(
            List<(string name, float[,] weights, float[,] gradients)> parameters,
            float clipValue)
        {
            foreach (var (_, _, gradients) in parameters)
            {
                for (int i = 0; i < gradients.GetLength(0); i++)
                {
                    for (int j = 0; j < gradients.GetLength(1); j++)
                    {
                        gradients[i, j] = Math.Clamp(gradients[i, j], -clipValue, clipValue);
                    }
                }
            }
        }
    }
}
