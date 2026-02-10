using System;
using System.Threading;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace NeuralNetwork.Cuda
{
    /// <summary>
    /// Monitors GPU memory usage and raises warnings before exhaustion.
    /// Queries cuMemGetInfo periodically to track free/total VRAM.
    /// Can be used to trigger GC or pool eviction when memory pressure is high.
    /// </summary>
    public class GpuMemoryWatchdog : IDisposable
    {
        private readonly CudaContext _context;
        private readonly bool _contextOwned;
        private readonly Timer _timer;
        private readonly float _warningThreshold;
        private readonly float _criticalThreshold;
        private bool _disposed;

        // Current state
        private long _freeBytes;
        private long _totalBytes;
        private bool _warningRaised;
        private bool _criticalRaised;

        /// <summary>Current free GPU memory in bytes.</summary>
        public long FreeBytes => Interlocked.Read(ref _freeBytes);

        /// <summary>Total GPU memory in bytes.</summary>
        public long TotalBytes => Interlocked.Read(ref _totalBytes);

        /// <summary>Current GPU memory usage as a fraction (0.0 - 1.0).</summary>
        public double UsageRatio
        {
            get
            {
                long total = TotalBytes;
                return total > 0 ? 1.0 - (double)FreeBytes / total : 0;
            }
        }

        /// <summary>Current free GPU memory in MB.</summary>
        public double FreeMB => FreeBytes / (1024.0 * 1024.0);

        /// <summary>Current used GPU memory in MB.</summary>
        public double UsedMB => (TotalBytes - FreeBytes) / (1024.0 * 1024.0);

        /// <summary>
        /// Fired when GPU memory usage exceeds the warning threshold.
        /// </summary>
        public event Action<GpuMemoryWarningEventArgs> OnWarning;

        /// <summary>
        /// Fired when GPU memory usage exceeds the critical threshold.
        /// </summary>
        public event Action<GpuMemoryWarningEventArgs> OnCritical;

        /// <summary>
        /// Create a GPU memory watchdog.
        /// </summary>
        /// <param name="context">CUDA context to monitor. Creates new one if null.</param>
        /// <param name="pollIntervalMs">How often to poll GPU memory (default: 5000ms).</param>
        /// <param name="warningThreshold">Usage ratio to trigger warning (default: 0.85 = 85%).</param>
        /// <param name="criticalThreshold">Usage ratio to trigger critical alert (default: 0.95 = 95%).</param>
        public GpuMemoryWatchdog(
            CudaContext context = null,
            int pollIntervalMs = 5000,
            float warningThreshold = 0.85f,
            float criticalThreshold = 0.95f)
        {
            _warningThreshold = warningThreshold;
            _criticalThreshold = criticalThreshold;

            if (context != null)
            {
                _context = context;
                _contextOwned = false;
            }
            else
            {
                _context = new CudaContext();
                _contextOwned = true;
            }

            // Take an initial reading
            PollMemory();

            // Start periodic monitoring
            _timer = new Timer(_ => PollMemory(), null, pollIntervalMs, pollIntervalMs);
        }

        /// <summary>
        /// Query current GPU memory status immediately (thread-safe).
        /// </summary>
        public (long freeBytes, long totalBytes) QueryMemory()
        {
            PollMemory();
            return (FreeBytes, TotalBytes);
        }

        /// <summary>
        /// Check if there is enough free GPU memory for the requested allocation.
        /// </summary>
        /// <param name="requiredBytes">Number of bytes needed.</param>
        /// <returns>True if sufficient memory is available.</returns>
        public bool HasSufficientMemory(long requiredBytes)
        {
            return FreeBytes > requiredBytes;
        }

        /// <summary>
        /// Throw an OutOfMemoryException if free GPU memory is below the requested amount.
        /// Call this before large allocations to fail fast with a clear message.
        /// </summary>
        public void EnsureSufficientMemory(long requiredBytes, string operationName = null)
        {
            PollMemory();
            if (FreeBytes < requiredBytes)
            {
                string op = operationName != null ? $" for '{operationName}'" : "";
                throw new OutOfMemoryException(
                    $"Insufficient GPU memory{op}. " +
                    $"Requested: {requiredBytes / (1024.0 * 1024.0):F1} MB, " +
                    $"Available: {FreeMB:F1} MB, " +
                    $"Total: {TotalBytes / (1024.0 * 1024.0):F1} MB " +
                    $"({UsageRatio:P1} used).");
            }
        }

        private void PollMemory()
        {
            try
            {
                // CudaContext.GetDeviceSize returns (free, total)
                var deviceMemory = _context.GetDeviceInfo().TotalGlobalMemory;
                SizeT free = 0;
                SizeT total = 0;
                DriverAPINativeMethods.MemoryManagement.cuMemGetInfo_v2(ref free, ref total);

                Interlocked.Exchange(ref _freeBytes, (long)free);
                Interlocked.Exchange(ref _totalBytes, (long)total);

                double usage = 1.0 - (double)free / (double)total;

                // Check critical threshold
                if (usage >= _criticalThreshold)
                {
                    if (!_criticalRaised)
                    {
                        _criticalRaised = true;
                        var args = new GpuMemoryWarningEventArgs(
                            (long)free, (long)total, usage, GpuMemorySeverity.Critical);
                        OnCritical?.Invoke(args);
                    }
                }
                else
                {
                    _criticalRaised = false;
                }

                // Check warning threshold
                if (usage >= _warningThreshold)
                {
                    if (!_warningRaised)
                    {
                        _warningRaised = true;
                        var args = new GpuMemoryWarningEventArgs(
                            (long)free, (long)total, usage, GpuMemorySeverity.Warning);
                        OnWarning?.Invoke(args);
                    }
                }
                else
                {
                    _warningRaised = false;
                }
            }
            catch
            {
                // Silently ignore polling failures — GPU may be busy
            }
        }

        /// <summary>
        /// Get a formatted status string.
        /// </summary>
        public string GetStatus()
        {
            return $"GPU Memory: {UsedMB:F1} / {TotalBytes / (1024.0 * 1024.0):F1} MB " +
                   $"({UsageRatio:P1} used, {FreeMB:F1} MB free)";
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _timer?.Dispose();

            if (_contextOwned)
            {
                _context?.Dispose();
            }

            GC.SuppressFinalize(this);
        }

        ~GpuMemoryWatchdog() => Dispose();
    }

    public enum GpuMemorySeverity
    {
        Warning,
        Critical
    }

    public class GpuMemoryWarningEventArgs
    {
        public long FreeBytes { get; }
        public long TotalBytes { get; }
        public double UsageRatio { get; }
        public GpuMemorySeverity Severity { get; }
        public double FreeMB => FreeBytes / (1024.0 * 1024.0);
        public double UsedMB => (TotalBytes - FreeBytes) / (1024.0 * 1024.0);

        public GpuMemoryWarningEventArgs(long freeBytes, long totalBytes, double usageRatio, GpuMemorySeverity severity)
        {
            FreeBytes = freeBytes;
            TotalBytes = totalBytes;
            UsageRatio = usageRatio;
            Severity = severity;
        }

        public override string ToString()
        {
            return $"[GPU {Severity}] Memory usage: {UsageRatio:P1} " +
                   $"({UsedMB:F1} MB used, {FreeMB:F1} MB free)";
        }
    }
}
