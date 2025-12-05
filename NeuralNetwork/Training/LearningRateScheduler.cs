using System;

namespace NeuralNetwork.Training
{
    /// <summary>
    /// Learning rate scheduler with warmup and decay strategies.
    /// Implements the schedule from "Attention Is All You Need":
    /// lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    /// </summary>
    public class LearningRateScheduler
    {
        private readonly float _baseLr;
        private readonly int _warmupSteps;
        private readonly ScheduleType _scheduleType;
        private readonly float _modelDim;
        private readonly float _minLr;
        private readonly int _totalSteps;

        private int _currentStep;

        public enum ScheduleType
        {
            /// <summary>
            /// Constant learning rate (no scheduling).
            /// </summary>
            Constant,

            /// <summary>
            /// Linear warmup followed by constant.
            /// </summary>
            LinearWarmup,

            /// <summary>
            /// Linear warmup followed by linear decay to min_lr.
            /// </summary>
            LinearWarmupLinearDecay,

            /// <summary>
            /// Linear warmup followed by cosine decay to min_lr.
            /// </summary>
            LinearWarmupCosineDecay,

            /// <summary>
            /// Transformer schedule: lr = d^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
            /// </summary>
            TransformerSchedule,

            /// <summary>
            /// Inverse square root decay after warmup.
            /// </summary>
            InverseSqrt
        }

        /// <summary>
        /// Current step number.
        /// </summary>
        public int CurrentStep => _currentStep;

        /// <summary>
        /// Current learning rate.
        /// </summary>
        public float CurrentLr { get; private set; }

        /// <summary>
        /// Create a learning rate scheduler.
        /// </summary>
        /// <param name="baseLr">Base/peak learning rate.</param>
        /// <param name="warmupSteps">Number of warmup steps.</param>
        /// <param name="scheduleType">Type of schedule to use.</param>
        /// <param name="totalSteps">Total training steps (required for decay schedules).</param>
        /// <param name="minLr">Minimum learning rate for decay schedules.</param>
        /// <param name="modelDim">Model dimension (for Transformer schedule).</param>
        public LearningRateScheduler(
            float baseLr = 1e-4f,
            int warmupSteps = 4000,
            ScheduleType scheduleType = ScheduleType.LinearWarmupCosineDecay,
            int totalSteps = 100000,
            float minLr = 1e-6f,
            int modelDim = 512)
        {
            _baseLr = baseLr;
            _warmupSteps = warmupSteps;
            _scheduleType = scheduleType;
            _totalSteps = totalSteps;
            _minLr = minLr;
            _modelDim = modelDim;
            _currentStep = 0;

            CurrentLr = GetLearningRate(0);
        }

        /// <summary>
        /// Advance one step and return the new learning rate.
        /// </summary>
        public float Step()
        {
            _currentStep++;
            CurrentLr = GetLearningRate(_currentStep);
            return CurrentLr;
        }

        /// <summary>
        /// Get learning rate for a specific step without advancing.
        /// </summary>
        public float GetLearningRate(int step)
        {
            return _scheduleType switch
            {
                ScheduleType.Constant => _baseLr,
                ScheduleType.LinearWarmup => LinearWarmup(step),
                ScheduleType.LinearWarmupLinearDecay => LinearWarmupLinearDecay(step),
                ScheduleType.LinearWarmupCosineDecay => LinearWarmupCosineDecay(step),
                ScheduleType.TransformerSchedule => TransformerSchedule(step),
                ScheduleType.InverseSqrt => InverseSqrtSchedule(step),
                _ => _baseLr
            };
        }

        /// <summary>
        /// Reset the scheduler to step 0.
        /// </summary>
        public void Reset()
        {
            _currentStep = 0;
            CurrentLr = GetLearningRate(0);
        }

        /// <summary>
        /// Set the current step (useful when resuming training).
        /// </summary>
        public void SetStep(int step)
        {
            _currentStep = step;
            CurrentLr = GetLearningRate(step);
        }

        #region Schedule Implementations

        private float LinearWarmup(int step)
        {
            if (step < _warmupSteps)
            {
                // Linear warmup from 0 to baseLr
                return _baseLr * (step + 1) / _warmupSteps;
            }
            return _baseLr;
        }

        private float LinearWarmupLinearDecay(int step)
        {
            if (step < _warmupSteps)
            {
                // Linear warmup
                return _baseLr * (step + 1) / _warmupSteps;
            }
            else
            {
                // Linear decay from baseLr to minLr
                int decaySteps = _totalSteps - _warmupSteps;
                int decayProgress = step - _warmupSteps;

                if (decayProgress >= decaySteps)
                    return _minLr;

                float decayRatio = (float)decayProgress / decaySteps;
                return _baseLr - (_baseLr - _minLr) * decayRatio;
            }
        }

        private float LinearWarmupCosineDecay(int step)
        {
            if (step < _warmupSteps)
            {
                // Linear warmup
                return _baseLr * (step + 1) / _warmupSteps;
            }
            else
            {
                // Cosine decay from baseLr to minLr
                int decaySteps = _totalSteps - _warmupSteps;
                int decayProgress = step - _warmupSteps;

                if (decayProgress >= decaySteps)
                    return _minLr;

                // Cosine decay: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
                float progress = (float)decayProgress / decaySteps;
                float cosineDecay = 0.5f * (1f + (float)Math.Cos(Math.PI * progress));
                return _minLr + (_baseLr - _minLr) * cosineDecay;
            }
        }

        private float TransformerSchedule(int step)
        {
            // From "Attention Is All You Need" paper
            // lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
            if (step == 0) step = 1;  // Avoid division by zero

            float factor = (float)Math.Pow(_modelDim, -0.5);
            float term1 = (float)Math.Pow(step, -0.5);
            float term2 = step * (float)Math.Pow(_warmupSteps, -1.5);

            return factor * Math.Min(term1, term2);
        }

        private float InverseSqrtSchedule(int step)
        {
            if (step < _warmupSteps)
            {
                // Linear warmup
                return _baseLr * (step + 1) / _warmupSteps;
            }
            else
            {
                // Inverse sqrt decay: lr = baseLr * sqrt(warmupSteps / step)
                return _baseLr * (float)Math.Sqrt((double)_warmupSteps / step);
            }
        }

        #endregion

        /// <summary>
        /// Create a scheduler with the original Transformer paper settings.
        /// </summary>
        public static LearningRateScheduler CreateTransformerScheduler(int modelDim, int warmupSteps = 4000)
        {
            return new LearningRateScheduler(
                baseLr: 1f,  // Not used directly in Transformer schedule
                warmupSteps: warmupSteps,
                scheduleType: ScheduleType.TransformerSchedule,
                modelDim: modelDim);
        }

        /// <summary>
        /// Create a common cosine decay scheduler.
        /// </summary>
        public static LearningRateScheduler CreateCosineScheduler(
            float peakLr,
            int warmupSteps,
            int totalSteps,
            float minLr = 1e-6f)
        {
            return new LearningRateScheduler(
                baseLr: peakLr,
                warmupSteps: warmupSteps,
                scheduleType: ScheduleType.LinearWarmupCosineDecay,
                totalSteps: totalSteps,
                minLr: minLr);
        }

        public override string ToString()
        {
            return $"LRScheduler(type={_scheduleType}, baseLr={_baseLr}, warmup={_warmupSteps}, " +
                   $"step={_currentStep}, currentLr={CurrentLr:E3})";
        }
    }
}
