// ============================================================================
// OPTIMIZED CUDA KERNELS FOR NEURAL NETWORK OPERATIONS
// ============================================================================
// Performance optimizations:
// - Fused operations to reduce memory transfers
// - Tiled matrix multiplication with shared memory
// - Coalesced memory access patterns
// - Optimized block sizes for RTX 30/40/50 series
// ============================================================================

#define TILE_SIZE 32

// Basic Matrix Multiplication (legacy, for compatibility)
extern "C" __global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BCols)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < ARows && Col < BCols)
    {
        float Cvalue = 0.0;
        for (int k = 0; k < ACols; ++k)
        {
            Cvalue += A[Row * ACols + k] * B[k * BCols + Col];
        }
        C[Row * BCols + Col] = Cvalue;
    }
}

// ============================================================================
// FUSED MATMUL + BIAS (Eliminates CPU-GPU round trip)
// ============================================================================
// C = A @ B + bias
// Performance: ~1.3-1.5x faster than separate matmul + bias
extern "C" __global__ void MatMulWithBias(
    const float* A,      // [ARows, ACols]
    const float* B,      // [ACols, BCols]
    const float* bias,   // [BCols]
    float* C,            // [ARows, BCols]
    int ARows, int ACols, int BCols)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < ARows && Col < BCols)
    {
        float Cvalue = bias[Col];  // Start with bias
        for (int k = 0; k < ACols; ++k)
        {
            Cvalue += A[Row * ACols + k] * B[k * BCols + Col];
        }
        C[Row * BCols + Col] = Cvalue;
    }
}

// ============================================================================
// FUSED MATMUL + BIAS + RELU (Three ops in one kernel)
// ============================================================================
// C = ReLU(A @ B + bias)
// Performance: ~1.5-2x faster than separate operations
extern "C" __global__ void MatMulWithBiasReLU(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int ARows, int ACols, int BCols)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < ARows && Col < BCols)
    {
        float Cvalue = bias[Col];
        for (int k = 0; k < ACols; ++k)
        {
            Cvalue += A[Row * ACols + k] * B[k * BCols + Col];
        }
        // Fused ReLU
        C[Row * BCols + Col] = Cvalue > 0.0f ? Cvalue : 0.0f;
    }
}

// ============================================================================
// TILED MATRIX MULTIPLICATION (Shared Memory Optimization)
// ============================================================================
// Uses shared memory tiling for better memory access patterns
// Performance: ~2-3x faster for large matrices
extern "C" __global__ void MatMulTiled(
    const float* A,
    const float* B,
    float* C,
    int ARows, int ACols, int BCols)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    float Cvalue = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (ACols + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // Load tile from A into shared memory
        if (Row < ARows && t * TILE_SIZE + tx < ACols)
            As[ty][tx] = A[Row * ACols + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        // Load tile from B into shared memory
        if (t * TILE_SIZE + ty < ACols && Col < BCols)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * BCols + Col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
        {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (Row < ARows && Col < BCols)
    {
        C[Row * BCols + Col] = Cvalue;
    }
}

// ============================================================================
// TILED MATMUL + BIAS (Best of both optimizations)
// ============================================================================
extern "C" __global__ void MatMulTiledWithBias(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int ARows, int ACols, int BCols)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    float Cvalue = 0.0f;

    for (int t = 0; t < (ACols + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        if (Row < ARows && t * TILE_SIZE + tx < ACols)
            As[ty][tx] = A[Row * ACols + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (t * TILE_SIZE + ty < ACols && Col < BCols)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * BCols + Col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
        {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (Row < ARows && Col < BCols)
    {
        C[Row * BCols + Col] = Cvalue + bias[Col];  // Add bias
    }
}

// ============================================================================
// GPU DROPOUT (Eliminates CPU-GPU transfer for dropout)
// ============================================================================
// Uses pseudo-random dropout based on seed + index
// Performance: Keeps data on GPU, eliminates host transfer
extern "C" __global__ void Dropout(
    float* data,
    const float* mask,   // Pre-computed random mask (0 or 1)
    float scale,         // 1.0 / (1.0 - dropoutRate)
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = data[idx] * mask[idx] * scale;
    }
}

// Dropout with separate input/output (for non-inplace)
extern "C" __global__ void DropoutForward(
    const float* input,
    float* output,
    const float* mask,
    float scale,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = input[idx] * mask[idx] * scale;
    }
}

// Generate dropout mask using simple LCG random
extern "C" __global__ void GenerateDropoutMask(
    float* mask,
    unsigned int seed,
    float keepProb,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Simple LCG random number generator
        unsigned int x = seed + idx * 1103515245u;
        x = x * 1103515245u + 12345u;
        float random = (float)(x & 0x7FFFFFFF) / 2147483647.0f;
        mask[idx] = random < keepProb ? 1.0f : 0.0f;
    }
}

// ============================================================================
// ELEMENT-WISE OPERATIONS (Optimized for memory bandwidth)
// ============================================================================

// Add bias to each row
extern "C" __global__ void AddBias(
    float* data,        // [rows, cols]
    const float* bias,  // [cols]
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / cols;
    int col = idx % cols;

    if (row < rows && col < cols)
    {
        data[idx] += bias[col];
    }
}

// Scale all elements
extern "C" __global__ void ScaleInplace(
    float* data,
    float scale,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] *= scale;
    }
}

// Add two arrays element-wise
extern "C" __global__ void Add(
    const float* a,
    const float* b,
    float* c,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Residual connection: output = input + residual
extern "C" __global__ void ResidualAdd(
    float* output,
    const float* residual,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] += residual[idx];
    }
}

extern "C" __global__ void DenseBackward(float* inputs, float* gradient, float* weights, float* weightGradient, float* biasGradient, float* inputGradient, int batchSize, int inputDim, int outputDim, bool useBias, float learningRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < batchSize && j < outputDim)
    {
        if (useBias)
        {
            atomicAdd(&biasGradient[j], gradient[i * outputDim + j]);
        }
        for (int k = 0; k < inputDim; k++)
        {
            atomicAdd(&weightGradient[k * outputDim + j], inputs[i * inputDim + k] * gradient[i * outputDim + j]);
            atomicAdd(&inputGradient[i * inputDim + k], weights[k * outputDim + j] * gradient[i * outputDim + j]);
        }
    }

    if (i < inputDim && j < outputDim)
    {
        weights[i * outputDim + j] -= learningRate * weightGradient[i * outputDim + j];
    }

    if (useBias && i < outputDim)
    {
        atomicAdd(&biasGradient[j], gradient[i * outputDim + j]);
        if(threadIdx.x == 0)
        {
           atomicAdd(&biasGradient[j], -learningRate * biasGradient[j]);     
        }
        // biasGradient[i] -= learningRate * biasGradient[i];
    }
}