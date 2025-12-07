using ManagedCuda;
using NeuralNetwork.Tests.Helpers;
using Xunit;

namespace NeuralNetwork.Tests.Cuda.TTS
{
    [Collection("CUDA")]
    public class TTSKernelLoadTests
    {
        private readonly CudaTestFixture _fixture;
        private readonly string _ptxPath;

        public TTSKernelLoadTests(CudaTestFixture fixture)
        {
            _fixture = fixture;
            string cuDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU");
            _ptxPath = Path.Combine(cuDir, "TTSKernel.ptx");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_PtxFileExists()
        {
            Assert.True(File.Exists(_ptxPath), $"TTSKernel.ptx not found at {_ptxPath}");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_CanLoadModule()
        {
            if (!File.Exists(_ptxPath))
            {
                Skip.If(true, "TTSKernel.ptx not found");
                return;
            }

            var module = _fixture.Context!.LoadModulePTX(_ptxPath);
            // Module is a value type (CUmodule), just verify no exception was thrown
            Assert.True(true, "Module loaded successfully");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_Conv1DForward_CanLoad()
        {
            VerifyKernelLoads("Conv1DForward");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_Conv1DForwardReLU_CanLoad()
        {
            VerifyKernelLoads("Conv1DForwardReLU");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_Conv1DForwardTanh_CanLoad()
        {
            VerifyKernelLoads("Conv1DForwardTanh");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_EmbeddingLookup_CanLoad()
        {
            VerifyKernelLoads("EmbeddingLookup");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_PrenetForward_CanLoad()
        {
            VerifyKernelLoads("PrenetForward");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_LocationSensitiveAttentionEnergy_CanLoad()
        {
            VerifyKernelLoads("LocationSensitiveAttentionEnergy");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_AttentionSoftmax_CanLoad()
        {
            VerifyKernelLoads("AttentionSoftmax");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_AttentionContext_CanLoad()
        {
            VerifyKernelLoads("AttentionContext");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_LSTMStep_CanLoad()
        {
            VerifyKernelLoads("LSTMStep");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_LSTMStepOptimized_CanLoad()
        {
            VerifyKernelLoads("LSTMStepOptimized");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_MelProjection_CanLoad()
        {
            VerifyKernelLoads("MelProjection");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_PostnetResidual_CanLoad()
        {
            VerifyKernelLoads("PostnetResidual");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_StopTokenSigmoid_CanLoad()
        {
            VerifyKernelLoads("StopTokenSigmoid");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_BatchedEmbeddingLookup_CanLoad()
        {
            VerifyKernelLoads("BatchedEmbeddingLookup");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_BatchedConv1DForward_CanLoad()
        {
            VerifyKernelLoads("BatchedConv1DForward");
        }

        [CudaFact]
        [Trait("Category", "CUDA")]
        [Trait("Category", "TTS")]
        public void TTSKernel_AllKernels_CanLoad()
        {
            if (!File.Exists(_ptxPath))
            {
                Skip.If(true, "TTSKernel.ptx not found");
                return;
            }

            var module = _fixture.Context!.LoadModulePTX(_ptxPath);

            string[] kernelNames = new[]
            {
                "Conv1DForward",
                "Conv1DForwardReLU",
                "Conv1DForwardTanh",
                "EmbeddingLookup",
                "PrenetForward",
                "LocationSensitiveAttentionEnergy",
                "AttentionSoftmax",
                "AttentionContext",
                "LSTMStep",
                "LSTMStepOptimized",
                "MelProjection",
                "PostnetResidual",
                "StopTokenSigmoid",
                "BatchedEmbeddingLookup",
                "BatchedConv1DForward"
            };

            var loadedKernels = new List<string>();
            var failedKernels = new List<string>();

            foreach (var name in kernelNames)
            {
                try
                {
                    var kernel = new CudaKernel(name, module, _fixture.Context);
                    loadedKernels.Add(name);
                }
                catch (Exception)
                {
                    failedKernels.Add(name);
                }
            }

            Assert.Empty(failedKernels);
            Assert.Equal(kernelNames.Length, loadedKernels.Count);
        }

        private void VerifyKernelLoads(string kernelName)
        {
            if (!File.Exists(_ptxPath))
            {
                Skip.If(true, "TTSKernel.ptx not found");
                return;
            }

            var module = _fixture.Context!.LoadModulePTX(_ptxPath);
            var kernel = new CudaKernel(kernelName, module, _fixture.Context);
            Assert.NotNull(kernel);
        }
    }
}
