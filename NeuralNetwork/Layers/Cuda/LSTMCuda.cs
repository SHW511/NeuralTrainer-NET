using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.VectorTypes;

namespace NeuralNetwork.Layers.Cuda
{
    public class LSTMCuda : Layer, IDisposable
    {
        public int Units { get; private set; }

        public float[,] _w;  // Weights and biases
        public float[,] _u;  // Weights and biases
        private float[] b;

        private CudaContext context;
        private CudaDeviceVariable<float> wDevice;
        private CudaDeviceVariable<float> uDevice;
        private CudaDeviceVariable<float> bDevice;
        private bool _disposed;

        // Cached linker image — JIT-compiled once in Build(), reused in every Call()
        private byte[] _cachedLinkerImage;

        public LSTMCuda(int units)
        {
            Units = units;
            _w = new float[0, 0]; // Initialize to avoid non-nullable warnings
            _u = new float[0, 0]; // Initialize to avoid non-nullable warnings
            b = new float[0];    // Initialize to avoid non-nullable warnings
            context = new CudaContext();
        }

        public override void Build(int[] inputShape)
        {
            int inputDim = inputShape[1];
            _w = Initializers.Initializers.GlorotUniform(inputDim, Units * 4);
            _u = Initializers.Initializers.GlorotUniform(Units, Units * 4);

            b = new float[Units * 4];

            wDevice = new CudaDeviceVariable<float>(_w.Length);
            uDevice = new CudaDeviceVariable<float>(_u.Length);
            bDevice = new CudaDeviceVariable<float>(b.Length);

            wDevice.CopyToDevice(_w);
            uDevice.CopyToDevice(_u);
            bDevice.CopyToDevice(b);

            // JIT-compile the linker image once at build time, not per forward pass
            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "LSTMKernelV2.ptx");
            CudaLinker linker = new CudaLinker();
            try
            {
                linker.AddFile(path, ManagedCuda.BasicTypes.CUJITInputType.PTX, null);

                if (Directory.Exists(@"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\"))
                {
                    linker.AddFile(@"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64\cudadevrt.lib", ManagedCuda.BasicTypes.CUJITInputType.Library, null);
                }
                else if (Directory.Exists(@"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\"))
                {
                    linker.AddFile(@"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64\cudadevrt.lib", ManagedCuda.BasicTypes.CUJITInputType.Library, null);
                }

                _cachedLinkerImage = linker.Complete();
            }
            finally
            {
                linker.Dispose();
            }

            Built = true;
        }

        public override float[,] Call(float[,] inputs)
        {
            int timesteps = inputs.GetLength(0);
            int inputDim = inputs.GetLength(1);

            float[,] h = new float[timesteps, Units];
            float[] c = new float[Units]; // Cell state
            float[] h_t = new float[Units]; // Hidden state

            // Allocate memory on the GPU with using statements to prevent leaks
            using var inputsDevice = new CudaDeviceVariable<float>(inputs.Length);
            using var hDevice = new CudaDeviceVariable<float>(h.Length);
            using var cDevice = new CudaDeviceVariable<float>(c.Length);
            using var h_tDevice = new CudaDeviceVariable<float>(h_t.Length);
            using var f_tDevice = new CudaDeviceVariable<float>(Units);
            using var i_tDevice = new CudaDeviceVariable<float>(Units);
            using var c_tildeDevice = new CudaDeviceVariable<float>(Units);
            using var o_tDevice = new CudaDeviceVariable<float>(Units);

            // Copy data to the GPU
            inputsDevice.CopyToDevice(inputs);
            cDevice.CopyToDevice(c);
            h_tDevice.CopyToDevice(h_t);

            // Use the cached linker image (JIT-compiled once in Build())
            var kernel = context.LoadKernelPTX(_cachedLinkerImage, "lstm_forward");

            // Define block and grid sizes
            dim3 blockSize = new dim3(Units);
            dim3 gridSize = new dim3(timesteps);

            // Launch the kernel
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;

            // Ensure memory allocation is successful
            if (inputsDevice.Size != inputs.Length || hDevice.Size != h.Length || cDevice.Size != c.Length || h_tDevice.Size != h_t.Length)
            {
                throw new InvalidOperationException("Memory allocation failed.");
            }

            kernel.Run(wDevice.DevicePointer, uDevice.DevicePointer, inputsDevice.DevicePointer, h_tDevice.DevicePointer, cDevice.DevicePointer, bDevice.DevicePointer,
                       f_tDevice.DevicePointer, i_tDevice.DevicePointer, c_tildeDevice.DevicePointer, o_tDevice.DevicePointer, cDevice.DevicePointer, hDevice.DevicePointer, inputDim, Units);

            // Copy the result back to the CPU
            hDevice.CopyToHost(h);

            return h;
        }

        public override float[,] Backward(float[,] gradient)
        {
            int timesteps = gradient.GetLength(0);
            int inputDim = _w.GetLength(0);

            float[,] dW = new float[inputDim, Units * 4];
            float[,] dU = new float[Units, Units * 4];
            float[] db = new float[Units * 4];
            float[,] dX = new float[timesteps, inputDim];

            // Allocate memory on the GPU with using statements to prevent leaks
            using var gradientDevice = new CudaDeviceVariable<float>(gradient.Length);
            using var dWDevice = new CudaDeviceVariable<float>(dW.Length);
            using var dUDevice = new CudaDeviceVariable<float>(dU.Length);
            using var dbDevice = new CudaDeviceVariable<float>(db.Length);
            using var dXDevice = new CudaDeviceVariable<float>(dX.Length);

            // Copy data to the GPU
            gradientDevice.CopyToDevice(gradient);

            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "LSTMKernel.ptx");

            // Load the kernel from the .cu file
            var kernel = context.LoadKernel(path, "LSTMBackward");

            // Define block and grid sizes
            dim3 blockSize = new dim3(128); //new dim3(Units);
            dim3 gridSize = new dim3(timesteps);

            // Launch the kernel
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;

            // Ensure memory allocation is successful
            if (gradientDevice.Size != gradient.Length || dWDevice.Size != dW.Length || dUDevice.Size != dU.Length || dbDevice.Size != db.Length || dXDevice.Size != dX.Length)
            {
                throw new InvalidOperationException("Memory allocation failed.");
            }

            kernel.Run(gradientDevice.DevicePointer, wDevice.DevicePointer, uDevice.DevicePointer, bDevice.DevicePointer, dWDevice.DevicePointer, dUDevice.DevicePointer, dbDevice.DevicePointer, dXDevice.DevicePointer, timesteps, inputDim, Units);

            // Copy the result back to the CPU
            dWDevice.CopyToHost(dW);
            dUDevice.CopyToHost(dU);
            dbDevice.CopyToHost(db);
            dXDevice.CopyToHost(dX);

            return dX;
        }

        public override int[] GetOutputShape(int[] inputShape)
        {
            return new int[] { inputShape[0], Units };
        }

        public override float[,,,] Call(float[,,,] inputs)
        {
            throw new NotImplementedException();
        }

        public override float[,,,] Backward(float[,,,] gradient)
        {
            throw new NotImplementedException();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            wDevice?.Dispose();
            uDevice?.Dispose();
            bDevice?.Dispose();
            context?.Dispose();

            GC.SuppressFinalize(this);
        }

        ~LSTMCuda() => Dispose();
    }
}
