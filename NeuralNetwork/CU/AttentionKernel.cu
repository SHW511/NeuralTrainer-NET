// AttentionKernel.cu - CUDA kernels for Transformer attention operations
// Implements scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

// Batched Matrix Multiplication: [B, M, K] x [B, K, N] -> [B, M, N]
// Each block handles one output tile for one batch element
extern "C" __global__ void BatchedMatMul(
    const float* A,      // [batch, M, K]
    const float* B,      // [batch, K, N]
    float* C,            // [batch, M, N]
    int batch,
    int M,
    int K,
    int N)
{
    int batchIdx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchIdx < batch && row < M && col < N)
    {
        int aOffset = batchIdx * M * K;
        int bOffset = batchIdx * K * N;
        int cOffset = batchIdx * M * N;

        float sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
            sum += A[aOffset + row * K + k] * B[bOffset + k * N + col];
        }
        C[cOffset + row * N + col] = sum;
    }
}

// Batched Matrix Multiplication with B transposed: [B, M, K] x [B, N, K]^T -> [B, M, N]
// Used for computing QK^T where K needs to be transposed
extern "C" __global__ void BatchedMatMulTransposeB(
    const float* A,      // [batch, M, K]
    const float* B,      // [batch, N, K] - will be transposed to [batch, K, N]
    float* C,            // [batch, M, N]
    int batch,
    int M,
    int K,
    int N)
{
    int batchIdx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchIdx < batch && row < M && col < N)
    {
        int aOffset = batchIdx * M * K;
        int bOffset = batchIdx * N * K;
        int cOffset = batchIdx * M * N;

        float sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
            // B is accessed as B[batch, col, k] which is transposed access
            sum += A[aOffset + row * K + k] * B[bOffset + col * K + k];
        }
        C[cOffset + row * N + col] = sum;
    }
}

// Batched Matrix Multiplication with A transposed: [B, K, M]^T x [B, K, N] -> [B, M, N]
// Used for backward pass
extern "C" __global__ void BatchedMatMulTransposeA(
    const float* A,      // [batch, K, M] - will be transposed to [batch, M, K]
    const float* B,      // [batch, K, N]
    float* C,            // [batch, M, N]
    int batch,
    int K,
    int M,
    int N)
{
    int batchIdx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchIdx < batch && row < M && col < N)
    {
        int aOffset = batchIdx * K * M;
        int bOffset = batchIdx * K * N;
        int cOffset = batchIdx * M * N;

        float sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
            // A is accessed as A[batch, k, row] which is transposed access
            sum += A[aOffset + k * M + row] * B[bOffset + k * N + col];
        }
        C[cOffset + row * N + col] = sum;
    }
}

// Scale tensor by a constant factor
extern "C" __global__ void Scale(
    float* data,         // [size] - in-place scaling
    float scale,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] *= scale;
    }
}

// Apply causal mask: set upper triangular elements (above diagonal) to -infinity
extern "C" __global__ void ApplyCausalMask(
    float* scores,       // [batch, seqLen, seqLen]
    int batch,
    int seqLen)
{
    int batchIdx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchIdx < batch && row < seqLen && col < seqLen)
    {
        // Causal mask: positions where col > row should be masked
        if (col > row)
        {
            int idx = batchIdx * seqLen * seqLen + row * seqLen + col;
            scores[idx] = -1e9f; // Use large negative instead of -inf for numerical stability
        }
    }
}

// Softmax along the last dimension for 3D tensor [batch, rows, cols]
// Each thread block processes one row
extern "C" __global__ void Softmax3D(
    const float* input,  // [batch, rows, cols]
    float* output,       // [batch, rows, cols]
    int batch,
    int rows,
    int cols)
{
    extern __shared__ float shared[];

    int batchIdx = blockIdx.y;
    int rowIdx = blockIdx.x;

    if (batchIdx >= batch || rowIdx >= rows) return;

    int baseIdx = batchIdx * rows * cols + rowIdx * cols;
    int tid = threadIdx.x;

    // Step 1: Find max value (parallel reduction)
    float localMax = -1e30f;
    for (int i = tid; i < cols; i += blockDim.x)
    {
        float val = input[baseIdx + i];
        if (val > localMax) localMax = val;
    }
    shared[tid] = localMax;
    __syncthreads();

    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (shared[tid + stride] > shared[tid])
                shared[tid] = shared[tid + stride];
        }
        __syncthreads();
    }
    float maxVal = shared[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    float localSum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
    {
        float expVal = expf(input[baseIdx + i] - maxVal);
        output[baseIdx + i] = expVal;
        localSum += expVal;
    }
    shared[tid] = localSum;
    __syncthreads();

    // Reduce to find global sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float sumVal = shared[0];
    __syncthreads();

    // Step 3: Normalize
    float invSum = 1.0f / sumVal;
    for (int i = tid; i < cols; i += blockDim.x)
    {
        output[baseIdx + i] *= invSum;
    }
}

// Softmax backward pass
// d_input = softmax_output * (d_output - sum(d_output * softmax_output))
extern "C" __global__ void SoftmaxBackward3D(
    const float* gradOutput,     // [batch, rows, cols]
    const float* softmaxOutput,  // [batch, rows, cols]
    float* gradInput,            // [batch, rows, cols]
    int batch,
    int rows,
    int cols)
{
    extern __shared__ float shared[];

    int batchIdx = blockIdx.y;
    int rowIdx = blockIdx.x;

    if (batchIdx >= batch || rowIdx >= rows) return;

    int baseIdx = batchIdx * rows * cols + rowIdx * cols;
    int tid = threadIdx.x;

    // Step 1: Compute dot product of gradOutput and softmaxOutput
    float localDot = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
    {
        localDot += gradOutput[baseIdx + i] * softmaxOutput[baseIdx + i];
    }
    shared[tid] = localDot;
    __syncthreads();

    // Reduce to find global dot product
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float dotProduct = shared[0];
    __syncthreads();

    // Step 2: Compute gradient
    for (int i = tid; i < cols; i += blockDim.x)
    {
        gradInput[baseIdx + i] = softmaxOutput[baseIdx + i] * (gradOutput[baseIdx + i] - dotProduct);
    }
}

// Transpose last two dimensions: [batch, M, N] -> [batch, N, M]
extern "C" __global__ void TransposeLast2D(
    const float* input,  // [batch, M, N]
    float* output,       // [batch, N, M]
    int batch,
    int M,
    int N)
{
    int batchIdx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchIdx < batch && row < M && col < N)
    {
        int inputIdx = batchIdx * M * N + row * N + col;
        int outputIdx = batchIdx * N * M + col * M + row;
        output[outputIdx] = input[inputIdx];
    }
}

// Element-wise multiply for dropout mask application
extern "C" __global__ void ElementwiseMultiply(
    const float* a,
    const float* b,
    float* c,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] * b[idx];
    }
}

// Element-wise add
extern "C" __global__ void ElementwiseAdd(
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

// Fused scaled dot-product attention forward pass
// Computes: output = softmax(Q @ K^T / scale) @ V
// This is a memory-efficient fused version
extern "C" __global__ void ScaledDotProductAttentionFused(
    const float* Q,          // [batch, seqLen, headDim]
    const float* K,          // [batch, seqLen, headDim]
    const float* V,          // [batch, seqLen, headDim]
    float* output,           // [batch, seqLen, headDim]
    float* attnWeights,      // [batch, seqLen, seqLen] - optional, for caching
    float scale,
    int batch,
    int seqLen,
    int headDim,
    bool useCausalMask,
    bool saveAttnWeights)
{
    // This kernel handles one (batch, query_pos) pair
    // It computes the full attention for that query position

    int batchIdx = blockIdx.y;
    int queryPos = blockIdx.x;
    int tid = threadIdx.x;

    if (batchIdx >= batch || queryPos >= seqLen) return;

    extern __shared__ float shared[];
    float* scores = shared;                          // [seqLen]
    float* values = shared + seqLen;                 // [blockDim.x] for reduction

    // Base offsets
    int qOffset = batchIdx * seqLen * headDim + queryPos * headDim;
    int kvOffset = batchIdx * seqLen * headDim;
    int attnOffset = batchIdx * seqLen * seqLen + queryPos * seqLen;
    int outOffset = batchIdx * seqLen * headDim + queryPos * headDim;

    // Step 1: Compute Q @ K^T for this query position
    // Each thread handles some key positions
    for (int keyPos = tid; keyPos < seqLen; keyPos += blockDim.x)
    {
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++)
        {
            dot += Q[qOffset + d] * K[kvOffset + keyPos * headDim + d];
        }
        dot *= scale;

        // Apply causal mask
        if (useCausalMask && keyPos > queryPos)
        {
            dot = -1e9f;
        }

        scores[keyPos] = dot;
    }
    __syncthreads();

    // Step 2: Softmax over scores
    // Find max
    float localMax = -1e30f;
    for (int i = tid; i < seqLen; i += blockDim.x)
    {
        if (scores[i] > localMax) localMax = scores[i];
    }
    values[tid] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride && values[tid + stride] > values[tid])
            values[tid] = values[tid + stride];
        __syncthreads();
    }
    float maxVal = values[0];
    __syncthreads();

    // Compute exp and sum
    float localSum = 0.0f;
    for (int i = tid; i < seqLen; i += blockDim.x)
    {
        float expVal = expf(scores[i] - maxVal);
        scores[i] = expVal;
        localSum += expVal;
    }
    values[tid] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    float sumVal = values[0];
    __syncthreads();

    // Normalize
    float invSum = 1.0f / sumVal;
    for (int i = tid; i < seqLen; i += blockDim.x)
    {
        scores[i] *= invSum;
        if (saveAttnWeights)
        {
            attnWeights[attnOffset + i] = scores[i];
        }
    }
    __syncthreads();

    // Step 3: Compute attention @ V for this query position
    // Each thread handles some output dimensions
    for (int d = tid; d < headDim; d += blockDim.x)
    {
        float sum = 0.0f;
        for (int keyPos = 0; keyPos < seqLen; keyPos++)
        {
            sum += scores[keyPos] * V[kvOffset + keyPos * headDim + d];
        }
        output[outOffset + d] = sum;
    }
}

// Backward pass for batched matmul: C = A @ B
// Computes dA = dC @ B^T and dB = A^T @ dC
extern "C" __global__ void BatchedMatMulBackwardA(
    const float* dC,     // [batch, M, N]
    const float* B,      // [batch, K, N]
    float* dA,           // [batch, M, K]
    int batch,
    int M,
    int K,
    int N)
{
    // dA = dC @ B^T
    int batchIdx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchIdx < batch && row < M && col < K)
    {
        int dCOffset = batchIdx * M * N;
        int bOffset = batchIdx * K * N;
        int dAOffset = batchIdx * M * K;

        float sum = 0.0f;
        for (int n = 0; n < N; n++)
        {
            // dC[row, n] * B[col, n] (B transposed access)
            sum += dC[dCOffset + row * N + n] * B[bOffset + col * N + n];
        }
        dA[dAOffset + row * K + col] = sum;
    }
}

extern "C" __global__ void BatchedMatMulBackwardB(
    const float* dC,     // [batch, M, N]
    const float* A,      // [batch, M, K]
    float* dB,           // [batch, K, N]
    int batch,
    int M,
    int K,
    int N)
{
    // dB = A^T @ dC
    int batchIdx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // K dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension

    if (batchIdx < batch && row < K && col < N)
    {
        int aOffset = batchIdx * M * K;
        int dCOffset = batchIdx * M * N;
        int dBOffset = batchIdx * K * N;

        float sum = 0.0f;
        for (int m = 0; m < M; m++)
        {
            // A[m, row] * dC[m, col] (A transposed access)
            sum += A[aOffset + m * K + row] * dC[dCOffset + m * N + col];
        }
        dB[dBOffset + row * N + col] = sum;
    }
}
