using ManagedCuda;
using Xunit;

namespace NeuralNetwork.Tests.Helpers
{
    /// <summary>
    /// Helper class for conditional test skipping.
    /// </summary>
    public static class Skip
    {
        /// <summary>
        /// Skips the test if the condition is true.
        /// Note: This uses xUnit's Assert.Skip which is available in xUnit 2.5+
        /// </summary>
        public static void If(bool condition, string reason)
        {
            if (condition)
            {
                Assert.Fail($"[SKIPPED] {reason}");
            }
        }
    }

    /// <summary>
    /// Shared CUDA context fixture for tests that require GPU.
    /// Use with IClassFixture<CudaTestFixture> to share context across tests.
    /// </summary>
    public class CudaTestFixture : IDisposable
    {
        public CudaContext? Context { get; private set; }
        public bool IsCudaAvailable { get; private set; }
        public string DeviceName { get; private set; } = "N/A";
        public string ComputeCapability { get; private set; } = "N/A";
        public long TotalMemoryBytes { get; private set; }

        public CudaTestFixture()
        {
            try
            {
                Context = new CudaContext();
                IsCudaAvailable = true;
                DeviceName = Context.GetDeviceName();
                ComputeCapability = Context.GetDeviceComputeCapability().ToString();
                TotalMemoryBytes = (long)Context.GetTotalDeviceMemorySize();
            }
            catch (Exception)
            {
                IsCudaAvailable = false;
                Context = null;
            }
        }

        /// <summary>
        /// Skip test if CUDA is not available.
        /// </summary>
        public void SkipIfNoCuda()
        {
            Skip.If(!IsCudaAvailable, "CUDA not available - skipping GPU test");
        }

        /// <summary>
        /// Get total GPU memory in GB.
        /// </summary>
        public double TotalMemoryGB => TotalMemoryBytes / (1024.0 * 1024.0 * 1024.0);

        public void Dispose()
        {
            Context?.Dispose();
            Context = null;
        }
    }

    /// <summary>
    /// Collection definition for sharing CUDA fixture across test classes.
    /// </summary>
    [CollectionDefinition("CUDA")]
    public class CudaCollection : ICollectionFixture<CudaTestFixture>
    {
        // This class has no code, and is never created. Its purpose is simply
        // to be the place to apply [CollectionDefinition] and all the
        // ICollectionFixture<> interfaces.
    }

    /// <summary>
    /// Attribute to mark tests that require CUDA.
    /// </summary>
    public class CudaFactAttribute : FactAttribute
    {
        public CudaFactAttribute()
        {
            // Check if CUDA is available
            try
            {
                using var ctx = new CudaContext();
            }
            catch
            {
                Skip = "CUDA not available";
            }
        }
    }

    /// <summary>
    /// Attribute to mark theory tests that require CUDA.
    /// </summary>
    public class CudaTheoryAttribute : TheoryAttribute
    {
        public CudaTheoryAttribute()
        {
            try
            {
                using var ctx = new CudaContext();
            }
            catch
            {
                Skip = "CUDA not available";
            }
        }
    }
}
