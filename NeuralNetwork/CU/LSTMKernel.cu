extern "C" __global__ void LSTMForward(float* inputs, float* w, float* u, float* b, float* h, float* c, float* h_t, int timesteps, int inputDim, int units)
{
    int t = blockIdx.x;
    int i = threadIdx.x;

    if (t < timesteps && i < units)
    {
        float* x_t = &inputs[t * inputDim];
        __shared__ float z[600]; // Assuming units * 4 = 600

        for (int j = 0; j < units * 4; j++)
        {
            z[j] = 0.0f;
            for (int k = 0; k < inputDim; k++)
            {
                z[j] += x_t[k] * w[k * units * 4 + j];
            }
            for (int k = 0; k < units; k++)
            {
                z[j] += h_t[k] * u[k * units * 4 + j];
            }
            z[j] += b[j];
        }

        float* i_gate = &z[0];
        float* f_gate = &z[units];
        float* o_gate = &z[units * 2];
        float* g_gate = &z[units * 3];

        for (int j = 0; j < units; j++)
        {
            i_gate[j] = 1.0f / (1.0f + exp(-i_gate[j]));
            f_gate[j] = 1.0f / (1.0f + exp(-f_gate[j]));
            o_gate[j] = 1.0f / (1.0f + exp(-o_gate[j]));
            g_gate[j] = tanh(g_gate[j]);
        }

        for (int j = 0; j < units; j++)
        {
            c[j] = f_gate[j] * c[j] + i_gate[j] * g_gate[j];
            h_t[j] = o_gate[j] * tanh(c[j]);
        }

        for (int j = 0; j < units; j++)
        {
            h[t * units + j] = h_t[j];
        }
    }
}

extern "C" __global__ void LSTMBackward(float* gradient, float* w, float* u, float* b, float* dW, float* dU, float* db, float* dX, int timesteps, int inputDim, int units)
{
    int t = blockIdx.x;
    int i = threadIdx.x;

    // Shared memory for intermediate calculations
    extern __shared__ float shared_mem[];
    float* z = shared_mem;
    float* i_gate = &z[0];
    float* f_gate = &z[units];
    float* o_gate = &z[units * 2];
    float* g_gate = &z[units * 3];

    // Initialize gradients
    if (t < timesteps && i < units)
    {
        float* x_t = &gradient[t * inputDim];
        float* c_t = &gradient[t * units];
        float* h_t = &gradient[t * units];

        // Forward pass to store cell states and hidden states
        for (int j = 0; j < units * 4; j++)
        {
            z[j] = 0.0f;
            for (int k = 0; k < inputDim; k++)
            {
                z[j] += x_t[k] * w[k * units * 4 + j];
            }
            for (int k = 0; k < units; k++)
            {
                z[j] += h_t[k] * u[k * units * 4 + j];
            }
            z[j] += b[j];
        }

        for (int j = 0; j < units; j++)
        {
            i_gate[j] = 1.0f / (1.0f + exp(-z[j]));
            f_gate[j] = 1.0f / (1.0f + exp(-z[units + j]));
            o_gate[j] = 1.0f / (1.0f + exp(-z[units * 2 + j]));
            g_gate[j] = tanh(z[units * 3 + j]);
        }

        // Backward pass
        __shared__ float dh_next[150]; // Assuming units = 150
        __shared__ float dc_next[150]; // Assuming units = 150
        for (int j = 0; j < units; j++)
        {
            dh_next[j] = 0.0f;
            dc_next[j] = 0.0f;
        }

        for (int t = timesteps - 1; t >= 0; t--)
        {
            float* dh = &gradient[t * units];
            for (int j = 0; j < units; j++)
            {
                dh[j] += dh_next[j];
            }

            float dc[150]; // Assuming units = 150
            for (int j = 0; j < units; j++)
            {
                dc[j] = dc_next[j] * f_gate[j] + dh[j] * o_gate[j] * (1 - tanh(c_t[j]) * tanh(c_t[j]));
            }

            for (int j = 0; j < units; j++)
            {
                atomicAdd(&db[j], dc[j]);
                for (int k = 0; k < inputDim; k++)
                {
                    atomicAdd(&dW[k * units * 4 + j], x_t[k] * dc[j]);
                }
                for (int k = 0; k < units; k++)
                {
                    atomicAdd(&dU[k * units * 4 + j], h_t[k] * dc[j]);
                }
            }

            for (int j = 0; j < units; j++)
            {
                dh_next[j] = dc[j];
                dc_next[j] = dc[j];
            }
        }
    }
}