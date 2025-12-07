using ManagedCuda;
using ManagedCuda.VectorTypes;
using NeuralNetwork.Tests.Helpers;
using Xunit;

namespace NeuralNetwork.Tests.Cuda.TTS
{
    [Collection("CUDA")]
    public class Conv1DKernelTests
    {
        private readonly CudaTestFixture _fixture;
        private readonly string _ptxPath;

        public Conv1DKernelTests(CudaTestFixture fixture)
        {
            _fixture = fixture;
            string cuDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU");
            _ptxPath = Path.Combine(cuDir, "TTSKernel.ptx");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void Conv1D_Forward_OutputShapeCorrect()
        {
            if (!File.Exists(_ptxPath))
            {
                Skip.If(true, "TTSKernel.ptx not found");
                return;
            }

            // Arrange
            int seqLen = 10;
            int inChannels = 4;
            int outChannels = 8;
            int kernelSize = 3;

            float[] input = TestDataBuilder.RandomVector(seqLen * inChannels);
            float[] weights = TestDataBuilder.RandomVector(kernelSize * inChannels * outChannels);
            float[] bias = TestDataBuilder.RandomVector(outChannels);
            float[] output = new float[seqLen * outChannels];

            var module = _fixture.Context!.LoadModulePTX(_ptxPath);
            var kernel = new CudaKernel("Conv1DForward", module, _fixture.Context);

            using var inputDevice = new CudaDeviceVariable<float>(input.Length);
            using var weightsDevice = new CudaDeviceVariable<float>(weights.Length);
            using var biasDevice = new CudaDeviceVariable<float>(bias.Length);
            using var outputDevice = new CudaDeviceVariable<float>(output.Length);

            inputDevice.CopyToDevice(input);
            weightsDevice.CopyToDevice(weights);
            biasDevice.CopyToDevice(bias);

            // Act
            kernel.BlockDimensions = new dim3(16, 16);
            kernel.GridDimensions = new dim3(
                (seqLen + 15) / 16,
                (outChannels + 15) / 16);

            kernel.Run(
                inputDevice.DevicePointer,
                weightsDevice.DevicePointer,
                biasDevice.DevicePointer,
                outputDevice.DevicePointer,
                seqLen,
                inChannels,
                outChannels,
                kernelSize);

            outputDevice.CopyToHost(output);

            // Assert
            MatrixAssert.HasLength(output, seqLen * outChannels);
            MatrixAssert.IsFinite(output);
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void Conv1D_Forward_MatchesCpuImplementation()
        {
            if (!File.Exists(_ptxPath))
            {
                Skip.If(true, "TTSKernel.ptx not found");
                return;
            }

            // Arrange
            int seqLen = 8;
            int inChannels = 4;
            int outChannels = 6;
            int kernelSize = 3;
            int padding = kernelSize / 2;

            // Use fixed seed for reproducibility
            var rng = new Random(42);
            float[] input = new float[seqLen * inChannels];
            float[] weights = new float[kernelSize * inChannels * outChannels];
            float[] bias = new float[outChannels];

            for (int i = 0; i < input.Length; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < weights.Length; i++)
                weights[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < bias.Length; i++)
                bias[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference implementation
            float[] cpuOutput = new float[seqLen * outChannels];
            for (int t = 0; t < seqLen; t++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    float sum = bias[oc];
                    for (int k = 0; k < kernelSize; k++)
                    {
                        int inputIdx = t + k - padding;
                        if (inputIdx >= 0 && inputIdx < seqLen)
                        {
                            for (int ic = 0; ic < inChannels; ic++)
                            {
                                int weightIdx = (k * inChannels + ic) * outChannels + oc;
                                sum += input[inputIdx * inChannels + ic] * weights[weightIdx];
                            }
                        }
                    }
                    cpuOutput[t * outChannels + oc] = sum;
                }
            }

            // GPU implementation
            var module = _fixture.Context!.LoadModulePTX(_ptxPath);
            var kernel = new CudaKernel("Conv1DForward", module, _fixture.Context);

            using var inputDevice = new CudaDeviceVariable<float>(input.Length);
            using var weightsDevice = new CudaDeviceVariable<float>(weights.Length);
            using var biasDevice = new CudaDeviceVariable<float>(bias.Length);
            using var outputDevice = new CudaDeviceVariable<float>(cpuOutput.Length);

            inputDevice.CopyToDevice(input);
            weightsDevice.CopyToDevice(weights);
            biasDevice.CopyToDevice(bias);

            kernel.BlockDimensions = new dim3(16, 16);
            kernel.GridDimensions = new dim3(
                (seqLen + 15) / 16,
                (outChannels + 15) / 16);

            kernel.Run(
                inputDevice.DevicePointer,
                weightsDevice.DevicePointer,
                biasDevice.DevicePointer,
                outputDevice.DevicePointer,
                seqLen,
                inChannels,
                outChannels,
                kernelSize);

            float[] gpuOutput = new float[seqLen * outChannels];
            outputDevice.CopyToHost(gpuOutput);

            // Assert - Compare CPU and GPU results
            MatrixAssert.AreAlmostEqual(cpuOutput, gpuOutput, tolerance: 1e-4f,
                message: "GPU Conv1D output should match CPU reference");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void Conv1DReLU_Forward_AppliesActivation()
        {
            if (!File.Exists(_ptxPath))
            {
                Skip.If(true, "TTSKernel.ptx not found");
                return;
            }

            // Arrange
            int seqLen = 8;
            int inChannels = 4;
            int outChannels = 6;
            int kernelSize = 3;

            // Use input that will produce some negative pre-activation values
            float[] input = TestDataBuilder.RandomVector(seqLen * inChannels, min: -2, max: 2);
            float[] weights = TestDataBuilder.RandomVector(kernelSize * inChannels * outChannels, min: -1, max: 1);
            float[] bias = TestDataBuilder.RandomVector(outChannels, min: -1, max: 1);
            float[] output = new float[seqLen * outChannels];

            var module = _fixture.Context!.LoadModulePTX(_ptxPath);
            var kernel = new CudaKernel("Conv1DForwardReLU", module, _fixture.Context);

            using var inputDevice = new CudaDeviceVariable<float>(input.Length);
            using var weightsDevice = new CudaDeviceVariable<float>(weights.Length);
            using var biasDevice = new CudaDeviceVariable<float>(bias.Length);
            using var outputDevice = new CudaDeviceVariable<float>(output.Length);

            inputDevice.CopyToDevice(input);
            weightsDevice.CopyToDevice(weights);
            biasDevice.CopyToDevice(bias);

            // Act
            kernel.BlockDimensions = new dim3(16, 16);
            kernel.GridDimensions = new dim3(
                (seqLen + 15) / 16,
                (outChannels + 15) / 16);

            kernel.Run(
                inputDevice.DevicePointer,
                weightsDevice.DevicePointer,
                biasDevice.DevicePointer,
                outputDevice.DevicePointer,
                seqLen,
                inChannels,
                outChannels,
                kernelSize);

            outputDevice.CopyToHost(output);

            // Assert - All outputs should be >= 0 (ReLU property)
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] >= 0, $"ReLU output at [{i}] should be >= 0, got {output[i]}");
            }
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void Conv1DTanh_Forward_OutputInRange()
        {
            if (!File.Exists(_ptxPath))
            {
                Skip.If(true, "TTSKernel.ptx not found");
                return;
            }

            // Arrange
            int seqLen = 8;
            int inChannels = 4;
            int outChannels = 6;
            int kernelSize = 3;

            float[] input = TestDataBuilder.RandomVector(seqLen * inChannels, min: -2, max: 2);
            float[] weights = TestDataBuilder.RandomVector(kernelSize * inChannels * outChannels, min: -1, max: 1);
            float[] bias = TestDataBuilder.RandomVector(outChannels, min: -1, max: 1);
            float[] output = new float[seqLen * outChannels];

            var module = _fixture.Context!.LoadModulePTX(_ptxPath);
            var kernel = new CudaKernel("Conv1DForwardTanh", module, _fixture.Context);

            using var inputDevice = new CudaDeviceVariable<float>(input.Length);
            using var weightsDevice = new CudaDeviceVariable<float>(weights.Length);
            using var biasDevice = new CudaDeviceVariable<float>(bias.Length);
            using var outputDevice = new CudaDeviceVariable<float>(output.Length);

            inputDevice.CopyToDevice(input);
            weightsDevice.CopyToDevice(weights);
            biasDevice.CopyToDevice(bias);

            // Act
            kernel.BlockDimensions = new dim3(16, 16);
            kernel.GridDimensions = new dim3(
                (seqLen + 15) / 16,
                (outChannels + 15) / 16);

            kernel.Run(
                inputDevice.DevicePointer,
                weightsDevice.DevicePointer,
                biasDevice.DevicePointer,
                outputDevice.DevicePointer,
                seqLen,
                inChannels,
                outChannels,
                kernelSize);

            outputDevice.CopyToHost(output);

            // Assert - All outputs should be in [-1, 1] (Tanh property)
            for (int i = 0; i < output.Length; i++)
            {
                Assert.True(output[i] >= -1 && output[i] <= 1,
                    $"Tanh output at [{i}] should be in [-1, 1], got {output[i]}");
            }
        }
    }
}
