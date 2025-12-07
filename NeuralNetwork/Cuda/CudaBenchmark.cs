using System;
using System.Diagnostics;
using ManagedCuda;
using ManagedCuda.VectorTypes;

namespace NeuralNetwork.Cuda
{
    /// <summary>
    /// Benchmark utilities to compare old vs optimized CUDA implementations.
    /// </summary>
    public static class CudaBenchmark
    {
        /// <summary>
        /// Run a comprehensive benchmark comparing old vs new implementations.
        /// </summary>
        public static void RunAll()
        {
            Console.WriteLine("=== CUDA Optimization Benchmark ===\n");

            // Check CUDA availability
            if (!CudaAccelerator.IsAvailable)
            {
                Console.WriteLine("CUDA not available. Skipping benchmark.");
                return;
            }

            var accelerator = CudaAccelerator.Default;
            Console.WriteLine($"Device: {accelerator.DeviceName}");
            Console.WriteLine($"Compute Capability: {accelerator.ComputeCapabilityMajor}.{accelerator.ComputeCapabilityMinor}");
            Console.WriteLine($"Memory: {accelerator.TotalMemory / (1024.0 * 1024.0 * 1024.0):F1} GB\n");

            // Benchmark memory pool
            BenchmarkMemoryPool(accelerator);

            // Benchmark kernel cache
            BenchmarkKernelCache(accelerator);

            // Benchmark matrix multiplication
            BenchmarkMatMul(accelerator);

            // Print final stats
            Console.WriteLine("\n" + accelerator.GetStats());
        }

        private static void BenchmarkMemoryPool(CudaAccelerator accelerator)
        {
            Console.WriteLine("--- Memory Pool Benchmark ---");
            const int iterations = 1000;
            const int bufferSize = 1024 * 1024; // 1M floats = 4MB

            var sw = Stopwatch.StartNew();

            // Without pooling (simulated)
            sw.Restart();
            for (int i = 0; i < iterations; i++)
            {
                using var buffer = new CudaDeviceVariable<float>(bufferSize);
            }
            var withoutPoolingMs = sw.ElapsedMilliseconds;

            // With pooling
            sw.Restart();
            for (int i = 0; i < iterations; i++)
            {
                using var buffer = accelerator.MemoryPool.RentFloat(bufferSize);
            }
            var withPoolingMs = sw.ElapsedMilliseconds;

            Console.WriteLine($"  Without pooling: {withoutPoolingMs} ms ({iterations} iterations)");
            Console.WriteLine($"  With pooling:    {withPoolingMs} ms ({iterations} iterations)");
            Console.WriteLine($"  Speedup: {(double)withoutPoolingMs / Math.Max(1, withPoolingMs):F1}x\n");
        }

        private static void BenchmarkKernelCache(CudaAccelerator accelerator)
        {
            Console.WriteLine("--- Kernel Cache Benchmark ---");
            const int iterations = 100;

            var kernelPath = accelerator.GetKernelPath("DenseKernel.ptx");
            if (!System.IO.File.Exists(kernelPath))
            {
                Console.WriteLine("  DenseKernel.ptx not found, skipping.\n");
                return;
            }

            var sw = Stopwatch.StartNew();

            // Without caching (simulated - reload each time)
            sw.Restart();
            for (int i = 0; i < iterations; i++)
            {
                var kernel = accelerator.Context.LoadKernelPTX(kernelPath, "MatMul");
            }
            var withoutCacheMs = sw.ElapsedMilliseconds;

            // With caching
            sw.Restart();
            for (int i = 0; i < iterations; i++)
            {
                var kernel = accelerator.KernelCache.GetKernel(kernelPath, "MatMul");
            }
            var withCacheMs = sw.ElapsedMilliseconds;

            Console.WriteLine($"  Without caching: {withoutCacheMs} ms ({iterations} iterations)");
            Console.WriteLine($"  With caching:    {withCacheMs} ms ({iterations} iterations)");
            Console.WriteLine($"  Speedup: {(double)withoutCacheMs / Math.Max(1, withCacheMs):F1}x\n");
        }

        private static void BenchmarkMatMul(CudaAccelerator accelerator)
        {
            Console.WriteLine("--- Matrix Multiplication Benchmark ---");
            const int M = 512, K = 512, N = 512;
            const int iterations = 100;

            var kernelPath = accelerator.GetKernelPath("DenseKernel.ptx");
            if (!System.IO.File.Exists(kernelPath))
            {
                Console.WriteLine("  DenseKernel.ptx not found, skipping.\n");
                return;
            }

            // Prepare data
            float[] A = new float[M * K];
            float[] B = new float[K * N];
            float[] C = new float[M * N];
            var rng = new Random(42);
            for (int i = 0; i < A.Length; i++) A[i] = (float)rng.NextDouble();
            for (int i = 0; i < B.Length; i++) B[i] = (float)rng.NextDouble();

            using var aDevice = accelerator.MemoryPool.RentFloat(M * K);
            using var bDevice = accelerator.MemoryPool.RentFloat(K * N);
            using var cDevice = accelerator.MemoryPool.RentFloat(M * N);

            aDevice.CopyToDevice(A);
            bDevice.CopyToDevice(B);

            var sw = Stopwatch.StartNew();

            // Basic MatMul
            var basicKernel = accelerator.KernelCache.GetKernel(kernelPath, "MatMul");
            dim3 blockSize = new dim3(16, 16);
            dim3 gridSize = new dim3((uint)((N + 15) / 16), (uint)((M + 15) / 16));

            sw.Restart();
            for (int i = 0; i < iterations; i++)
            {
                basicKernel.BlockDimensions = blockSize;
                basicKernel.GridDimensions = gridSize;
                basicKernel.Run(aDevice.DevicePointer, bDevice.DevicePointer, cDevice.DevicePointer, M, K, N);
            }
            accelerator.Synchronize();
            var basicMs = sw.ElapsedMilliseconds;

            // Tiled MatMul (if available)
            long tiledMs = basicMs;
            try
            {
                var tiledKernel = accelerator.KernelCache.GetKernel(kernelPath, "MatMulTiled");
                dim3 tiledBlockSize = new dim3(32, 32);
                dim3 tiledGridSize = new dim3((uint)((N + 31) / 32), (uint)((M + 31) / 32));

                sw.Restart();
                for (int i = 0; i < iterations; i++)
                {
                    tiledKernel.BlockDimensions = tiledBlockSize;
                    tiledKernel.GridDimensions = tiledGridSize;
                    tiledKernel.Run(aDevice.DevicePointer, bDevice.DevicePointer, cDevice.DevicePointer, M, K, N);
                }
                accelerator.Synchronize();
                tiledMs = sw.ElapsedMilliseconds;
            }
            catch
            {
                Console.WriteLine("  Tiled kernel not available (PTX needs recompilation)");
            }

            double gflops = (2.0 * M * N * K * iterations) / (basicMs * 1e6);
            double tiledGflops = (2.0 * M * N * K * iterations) / (tiledMs * 1e6);

            Console.WriteLine($"  Matrix size: {M}x{K} @ {K}x{N}");
            Console.WriteLine($"  Basic MatMul:  {basicMs} ms ({gflops:F1} GFLOPS)");
            Console.WriteLine($"  Tiled MatMul:  {tiledMs} ms ({tiledGflops:F1} GFLOPS)");
            Console.WriteLine($"  Speedup: {(double)basicMs / Math.Max(1, tiledMs):F2}x\n");
        }

        /// <summary>
        /// Quick sanity check that the optimized infrastructure works.
        /// </summary>
        public static bool QuickTest()
        {
            try
            {
                if (!CudaAccelerator.IsAvailable)
                {
                    Console.WriteLine("CUDA not available");
                    return false;
                }

                var accelerator = CudaAccelerator.Default;

                // Test memory pool
                using var buffer1 = accelerator.MemoryPool.RentFloat(1024);
                using var buffer2 = accelerator.MemoryPool.RentFloat(1024);

                float[] testData = new float[1024];
                for (int i = 0; i < testData.Length; i++) testData[i] = i;

                buffer1.CopyToDevice(testData);
                buffer1.CopyToHost(testData);

                // Test kernel cache
                var kernelPath = accelerator.GetKernelPath("DenseKernel.ptx");
                if (System.IO.File.Exists(kernelPath))
                {
                    var kernel = accelerator.KernelCache.GetKernel(kernelPath, "MatMul");
                }

                // Test stream manager
                using var stream = accelerator.StreamManager.RentStream();
                stream.Synchronize();

                Console.WriteLine("Quick test passed!");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Quick test failed: {ex.Message}");
                return false;
            }
        }
    }
}
