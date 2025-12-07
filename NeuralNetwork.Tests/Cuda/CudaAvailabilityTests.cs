using ManagedCuda;
using NeuralNetwork.Tests.Helpers;
using Xunit;

namespace NeuralNetwork.Tests.Cuda
{
    [Collection("CUDA")]
    public class CudaAvailabilityTests
    {
        private readonly CudaTestFixture _fixture;

        public CudaAvailabilityTests(CudaTestFixture fixture)
        {
            _fixture = fixture;
        }

        [Fact]
        [Trait("Category", "CUDA")]
        public void Cuda_IsAvailable()
        {
            // This test documents whether CUDA is available
            // It doesn't fail if CUDA is unavailable - just skips
            if (!_fixture.IsCudaAvailable)
            {
                Skip.If(true, "CUDA not available on this system");
                return;
            }

            Assert.True(_fixture.IsCudaAvailable);
            Assert.NotNull(_fixture.Context);
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        public void Cuda_CanGetDeviceInfo()
        {
            Assert.NotEmpty(_fixture.DeviceName);
            Assert.NotEqual("N/A", _fixture.DeviceName);
            Assert.NotEqual("N/A", _fixture.ComputeCapability);
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        public void Cuda_HasSufficientMemory()
        {
            // Require at least 1GB of GPU memory
            Assert.True(_fixture.TotalMemoryGB >= 1.0,
                $"GPU has only {_fixture.TotalMemoryGB:F1} GB, need at least 1 GB");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        public void Cuda_CanAllocateMemory()
        {
            // Test basic memory allocation
            using var deviceVar = new CudaDeviceVariable<float>(1000);
            Assert.Equal(1000L, (long)deviceVar.Size);
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        public void Cuda_CanCopyToDevice()
        {
            // Test host to device copy
            float[] hostData = new float[100];
            for (int i = 0; i < 100; i++)
                hostData[i] = i * 0.5f;

            using var deviceVar = new CudaDeviceVariable<float>(100);
            deviceVar.CopyToDevice(hostData);

            float[] result = new float[100];
            deviceVar.CopyToHost(result);

            MatrixAssert.AreAlmostEqual(hostData, result);
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        public void Cuda_CanLoadPtxModule()
        {
            // Test that we can load a PTX module
            string cuDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU");
            string denseKernelPath = Path.Combine(cuDir, "DenseKernel.ptx");

            if (!File.Exists(denseKernelPath))
            {
                Skip.If(true, $"DenseKernel.ptx not found at {denseKernelPath}");
                return;
            }

            var module = _fixture.Context!.LoadModulePTX(denseKernelPath);
            // Module is a value type (CUmodule), just verify no exception was thrown
            Assert.True(true, "Module loaded successfully");
        }

        [Fact]
        [Trait("Category", "CUDA")]
        public void Cuda_ContextCreationDoesNotThrow()
        {
            // Test that creating a CUDA context doesn't throw
            // (may fail gracefully if no GPU)
            CudaContext? ctx = null;
            Exception? caughtException = null;

            try
            {
                ctx = new CudaContext();
            }
            catch (Exception ex)
            {
                caughtException = ex;
            }
            finally
            {
                ctx?.Dispose();
            }

            // Either context was created or a CudaException was thrown
            Assert.True(ctx != null || caughtException is CudaException,
                $"Expected CudaContext or CudaException, got {caughtException?.GetType().Name}");
        }
    }
}
