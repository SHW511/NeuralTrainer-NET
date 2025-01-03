extern "C" __global__ void GRUForward(float* inputs, float* w, float* u, float* b, float* h, float* h_t, int timesteps, int inputDim, int units)
{
    int t = blockIdx.x;
    int i = threadIdx.x;

    if (t < timesteps && i < units)
    {
        float* x_t = &inputs[t * inputDim];
        __shared__ float z[600]; // Assuming units * 3 = 600

        for (int j = 0; j < units * 3; j++)
        {
            z[j] = 0.0f;
            for (int k = 0; k < inputDim; k++)
            {
                z[j] += x_t[k] * w[k * units * 3 + j];
            }
            for (int k = 0; k < units; k++)
            {
                z[j] += h_t[k] * u[k * units * 3 + j];
            }
            z[j] += b[j];
        }

        float* r_gate = &z[0];
        float* z_gate = &z[units];
        float* h_hat = &z[units * 2];

        for (int j = 0; j < units; j++)
        {
            r_gate[j] = 1.0f / (1.0f + exp(-r_gate[j]));
            z_gate[j] = 1.0f / (1.0f + exp(-z_gate[j]));
            h_hat[j] = tanh(h_hat[j]);
        }

        for (int j = 0; j < units; j++)
        {
            h_t[j] = (1 - z_gate[j]) * h_t[j] + z_gate[j] * h_hat[j];
        }

        for (int j = 0; j < units; j++)
        {
            h[t * units + j] = h_t[j];
        }
    }
}

extern "C" __global__ void GRUBackward(float* gradient, float* w, float* u, float* b, float* dW, float* dU, float* db, float* dX, int timesteps, int inputDim, int units)
{
    int t = blockIdx.x;
    int i = threadIdx.x;

    // Shared memory for intermediate calculations
    extern __shared__ float shared_mem[];
    float* z = shared_mem;
    float* r_gate = &z[0];
    float* z_gate = &z[units];
    float* h_hat = &z[units * 2];

    // Initialize gradients
    if (t < timesteps && i < units)
    {
        float* x_t = &gradient[t * inputDim];
        float* h_t = &gradient[t * units];

        // Forward pass to store gate activations
        for (int j = 0; j < units * 3; j++)
        {
            z[j] = 0.0f;
            for (int k = 0; k < inputDim; k++)
            {
                z[j] += x_t[k] * w[k * units * 3 + j];
            }
            for (int k = 0; k < units; k++)
            {
                z[j] += h_t[k] * u[k * units * 3 + j];
            }
            z[j] += b[j];
        }

        for (int j = 0; j < units; j++)
        {
            r_gate[j] = 1.0f / (1.0f + exp(-z[j]));
            z_gate[j] = 1.0f / (1.0f + exp(-z[units + j]));
            h_hat[j] = tanh(z[units * 2 + j]);
        }

        // Backward pass
        __shared__ float dh_next[150]; // Assuming units = 150
        for (int j = 0; j < units; j++)
        {
            dh_next[j] = 0.0f;
        }

        for (int t = timesteps - 1; t >= 0; t--)
        {
            float* dh = &gradient[t * units];
            for (int j = 0; j < units; j++)
            {
                dh[j] += dh_next[j];
            }

            float dz_gate[150]; // Assuming units = 150
            float dr_gate[150]; // Assuming units = 150
            float dh_hat[150];  // Assuming units = 150

            for (int j = 0; j < units; j++)
            {
                dz_gate[j] = dh[j] * (h_hat[j] - h_t[j]) * z_gate[j] * (1 - z_gate[j]);
                dh_hat[j] = dh[j] * z_gate[j] * (1 - h_hat[j] * h_hat[j]);
                dr_gate[j] = dh_hat[j] * h_t[j] * r_gate[j] * (1 - r_gate[j]);
            }

            for (int j = 0; j < units; j++)
            {
                atomicAdd(&db[j], dr_gate[j]);
                atomicAdd(&db[units + j], dz_gate[j]);
                atomicAdd(&db[units * 2 + j], dh_hat[j]);

                for (int k = 0; k < inputDim; k++)
                {
                    atomicAdd(&dW[k * units * 3 + j], x_t[k] * dr_gate[j]);
                    atomicAdd(&dW[k * units * 3 + units + j], x_t[k] * dz_gate[j]);
                    atomicAdd(&dW[k * units * 3 + units * 2 + j], x_t[k] * dh_hat[j]);
                }

                for (int k = 0; k < units; k++)
                {
                    atomicAdd(&dU[k * units * 3 + j], h_t[k] * dr_gate[j]);
                    atomicAdd(&dU[k * units * 3 + units + j], h_t[k] * dz_gate[j]);
                    atomicAdd(&dU[k * units * 3 + units * 2 + j], h_t[k] * dh_hat[j]);
                }
            }

            for (int j = 0; j < units; j++)
            {
                dh_next[j] = dr_gate[j] + dz_gate[j] + dh_hat[j];
            }
        }
    }
}