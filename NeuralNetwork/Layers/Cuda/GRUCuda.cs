using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda.VectorTypes;
using ManagedCuda;

namespace NeuralNetwork.Layers.Cuda
{
    public class GRUCuda : Layer
    {
        public int Units { get; private set; }

        public float[,] _w;  // Weights and biases
        public float[,] _u;  // Weights and biases
        private float[] b;

        private CudaContext context;
        private bool _contextOwned;
        private CudaDeviceVariable<float> wDevice;
        private CudaDeviceVariable<float> uDevice;
        private CudaDeviceVariable<float> bDevice;
        private bool _disposed;

        // Cached kernel path and kernels
        private string _kernelPath;
        private CudaKernel _forwardKernel;
        private CudaKernel _backwardKernel;

        public GRUCuda(int units, CudaContext context = null)
        {
            Units = units;
            _w = new float[0, 0]; // Initialize to avoid non-nullable warnings
            _u = new float[0, 0]; // Initialize to avoid non-nullable warnings
            b = new float[0];    // Initialize to avoid non-nullable warnings

            if (context != null)
            {
                this.context = context;
                _contextOwned = false;
            }
            else
            {
                this.context = new CudaContext();
                _contextOwned = true;
            }

            _kernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "CU", "GRUKernel.ptx");
        }

        public override void Build(int[] inputShape)
        {
            int inputDim = inputShape[1];
            _w = Initializers.Initializers.GlorotUniform(inputDim, Units * 3);
            _u = Initializers.Initializers.GlorotUniform(Units, Units * 3);

            b = new float[Units * 3];

            wDevice = new CudaDeviceVariable<float>(_w.Length);
            uDevice = new CudaDeviceVariable<float>(_u.Length);
            bDevice = new CudaDeviceVariable<float>(b.Length);

            wDevice.CopyToDevice(_w);
            uDevice.CopyToDevice(_u);
            bDevice.CopyToDevice(b);

            // Pre-load and cache kernels
            _forwardKernel = context.LoadKernel(_kernelPath, "GRUForward");
            _backwardKernel = context.LoadKernel(_kernelPath, "GRUBackward");

            Built = true;
        }

        public override float[,] Call(float[,] inputs)
        {
            int timesteps = inputs.GetLength(0);
            int inputDim = inputs.GetLength(1);

            float[,] h = new float[timesteps, Units];
            float[] h_t = new float[Units]; // Hidden state

            // Allocate memory on the GPU with using statements to prevent leaks
            using var inputsDevice = new CudaDeviceVariable<float>(inputs.Length);
            using var hDevice = new CudaDeviceVariable<float>(h.Length);
            using var h_tDevice = new CudaDeviceVariable<float>(h_t.Length);

            // Copy data to the GPU
            inputsDevice.CopyToDevice(inputs);
            h_tDevice.CopyToDevice(h_t);

            // Use cached forward kernel
            var kernel = _forwardKernel;

            // Define block and grid sizes
            dim3 blockSize = new dim3(Units);
            dim3 gridSize = new dim3(timesteps);

            // Launch the kernel
            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;

            // Ensure memory allocation is successful
            if (inputsDevice.Size != inputs.Length || hDevice.Size != h.Length || h_tDevice.Size != h_t.Length)
            {
                throw new InvalidOperationException("Memory allocation failed.");
            }

            kernel.Run(inputsDevice.DevicePointer, wDevice.DevicePointer, uDevice.DevicePointer, bDevice.DevicePointer, hDevice.DevicePointer, h_tDevice.DevicePointer, timesteps, inputDim, Units);

            // Copy the result back to the CPU
            hDevice.CopyToHost(h);

            return h;
        }

        public override float[,] Backward(float[,] gradient)
        {
            int timesteps = gradient.GetLength(0);
            int inputDim = _w.GetLength(0);

            float[,] dW = new float[inputDim, Units * 3];
            float[,] dU = new float[Units, Units * 3];
            float[] db = new float[Units * 3];
            float[,] dX = new float[timesteps, inputDim];

            // Allocate memory on the GPU with using statements to prevent leaks
            using var gradientDevice = new CudaDeviceVariable<float>(gradient.Length);
            using var dWDevice = new CudaDeviceVariable<float>(dW.Length);
            using var dUDevice = new CudaDeviceVariable<float>(dU.Length);
            using var dbDevice = new CudaDeviceVariable<float>(db.Length);
            using var dXDevice = new CudaDeviceVariable<float>(dX.Length);

            // Copy data to the GPU
            gradientDevice.CopyToDevice(gradient);

            // Use cached backward kernel
            var kernel = _backwardKernel;

            // Define block and grid sizes
            dim3 blockSize = new dim3(Units);
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

        public override void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            wDevice?.Dispose();
            uDevice?.Dispose();
            bDevice?.Dispose();

            if (_contextOwned)
            {
                context?.Dispose();
            }

            GC.SuppressFinalize(this);
        }

        ~GRUCuda() => Dispose();
    }
}
