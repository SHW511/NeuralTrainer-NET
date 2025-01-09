#include <cuda_runtime.h>
#include <math.h>

__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
        {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void compute_gates(const float *Wx, const float *Wh, const float *x_t, const float *h_prev, const float *b, float *f_t, float *i_t, float *c_tilde, float *o_t, int input_size, int hidden_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < hidden_size)
    {
        float Wx_combined = 0.0f;
        float Wh_combined = 0.0f;

        for (int i = 0; i < input_size; i++)
        {
            Wx_combined += Wx[idx * input_size + i] * x_t[i];
        }

        for (int i = 0; i < hidden_size; i++)
        {
            Wh_combined += Wh[idx * hidden_size + i] * h_prev[i];
        }

        float z = Wx_combined + Wh_combined + b[idx];

        if (idx < hidden_size)
            f_t[idx] = sigmoid(z);

        if (idx >= hidden_size && idx < 2 * hidden_size)
            i_t[idx - hidden_size] = sigmoid(z);

        if (idx >= 2 * hidden_size && idx < 3 * hidden_size)
            c_tilde[idx - 2 * hidden_size] = tanh(z);

        if (idx >= 3 * hidden_size)
            o_t[idx - 3 * hidden_size] = sigmoid(z);
    }
}

__global__ void update_states(const float *f_t, const float *i_t, const float *c_tilde, const float *o_t, const float *c_prev, float *c_t, float *h_t, int hidden_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < hidden_size)
    {
        c_t[idx] = f_t[idx] * c_prev[idx] + i_t[idx] * c_tilde[idx];
        h_t[idx] = o_t[idx] * tanh(c_t[idx]);
    }
}

__global__ void reverse_sequence(const float *input, float *reversed, int sequence_length, int input_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = sequence_length * input_size;

    if (idx < total_size)
    {
        int seq_idx = idx / input_size;
        int feature_idx = idx % input_size;
        int reversed_idx = (sequence_length - 1 - seq_idx) * input_size + feature_idx;
        reversed[reversed_idx] = input[idx];
    }
}

extern "C" __global__ void lstm_forward(const float *Wx, const float *Wh, const float *x_t, const float *h_prev, const float *c_prev, const float *b, float *f_t, float *i_t, float *c_tilde,
                                        float *o_t, float *c_t, float *h_t, int input_size, int hidden_size)
{
    dim3 blockDim(256);
    dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x);

    compute_gates<<<gridDim, blockDim>>>(Wx, Wh, x_t, h_prev, b, f_t, i_t, c_tilde, o_t, input_size, hidden_size);

    update_states<<<gridDim, blockDim>>>(f_t, i_t, c_tilde, o_t, c_prev, c_t, h_t, hidden_size);

    // cudaDeviceSynchronize();
}

extern "C" __global__ void lstm_backward(const float *Wx, const float *Wh, const float *x_t, const float *h_prev,
                              const float *c_prev, const float *b, float *f_t, float *i_t, float *c_tilde,
                              float *o_t, float *c_t, float *h_t, int input_size, int hidden_size, int sequence_length)
{
    float* d_reversed_x_t;
    cudaMalloc((void**)&d_reversed_x_t, sequence_length * input_size * sizeof(float));

    int total_size = sequence_length * input_size;
    dim3 reverseBlockDim(256);
    dim3 reverseGridDim((total_size + reverseBlockDim.x - 1) / reverseBlockDim.x);
    reverse_sequence<<<reverseGridDim, reverseBlockDim>>>(x_t, d_reversed_x_t, sequence_length, input_size);

    for (int t = sequence_length; t >= 0; --t)
    {
        const float* x_curr = &d_reversed_x_t[t * input_size];
        const float* h_prev_curr = &h_prev[t * hidden_size];
        const float* c_prev_curr = &c_prev[t * hidden_size];

        float* h_t_curr = &h_t[t * hidden_size];
        float* c_t_curr = &c_t[t * hidden_size];

        compute_gates<<<reverseGridDim, reverseBlockDim>>>(Wx, Wh, x_curr, h_prev_curr, b, f_t, i_t, c_tilde, o_t, input_size, hidden_size);
        update_states<<<reverseGridDim, reverseBlockDim>>>(f_t, i_t, c_tilde, o_t, c_prev_curr, c_t_curr, h_t_curr, hidden_size);
    }

    cudaFree(d_reversed_x_t);

    // cudaDeviceSynchronize();
    
}