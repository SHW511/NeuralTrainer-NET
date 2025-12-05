// LayerNormKernel.cu - CUDA kernels for Layer Normalization
// Normalizes across the last dimension (features) for each sample

// Forward pass: For each row, compute mean, variance, normalize, then scale+shift
// Uses Welford's online algorithm for numerical stability
extern "C" __global__ void LayerNormForward(
    const float* input,      // [batch, features]
    const float* gamma,      // [features] - scale parameter
    const float* beta,       // [features] - shift parameter
    float* output,           // [batch, features]
    float* mean,             // [batch] - cached for backward
    float* variance,         // [batch] - cached for backward
    float* normalized,       // [batch, features] - cached for backward
    int batch,
    int features,
    float epsilon)
{
    extern __shared__ float shared[];

    int batchIdx = blockIdx.x;
    int tid = threadIdx.x;

    if (batchIdx >= batch) return;

    int baseIdx = batchIdx * features;

    // Step 1: Compute mean using parallel reduction
    float localSum = 0.0f;
    for (int i = tid; i < features; i += blockDim.x)
    {
        localSum += input[baseIdx + i];
    }
    shared[tid] = localSum;
    __syncthreads();

    // Reduce to find sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float meanVal = shared[0] / features;
    if (tid == 0) mean[batchIdx] = meanVal;
    __syncthreads();

    // Step 2: Compute variance using parallel reduction
    float localVarSum = 0.0f;
    for (int i = tid; i < features; i += blockDim.x)
    {
        float diff = input[baseIdx + i] - meanVal;
        localVarSum += diff * diff;
    }
    shared[tid] = localVarSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float varVal = shared[0] / features;
    if (tid == 0) variance[batchIdx] = varVal;
    __syncthreads();

    // Step 3: Normalize and apply scale/shift
    float stdInv = rsqrtf(varVal + epsilon);
    for (int i = tid; i < features; i += blockDim.x)
    {
        float norm = (input[baseIdx + i] - meanVal) * stdInv;
        normalized[baseIdx + i] = norm;
        output[baseIdx + i] = gamma[i] * norm + beta[i];
    }
}

// Forward pass for 3D tensors [batch, seqLen, features]
// Treats batch*seqLen as the batch dimension for normalization
extern "C" __global__ void LayerNormForward3D(
    const float* input,      // [batch * seqLen, features] (flattened)
    const float* gamma,      // [features]
    const float* beta,       // [features]
    float* output,           // [batch * seqLen, features]
    float* mean,             // [batch * seqLen]
    float* variance,         // [batch * seqLen]
    float* normalized,       // [batch * seqLen, features]
    int totalRows,           // batch * seqLen
    int features,
    float epsilon)
{
    extern __shared__ float shared[];

    int rowIdx = blockIdx.x;
    int tid = threadIdx.x;

    if (rowIdx >= totalRows) return;

    int baseIdx = rowIdx * features;

    // Step 1: Compute mean
    float localSum = 0.0f;
    for (int i = tid; i < features; i += blockDim.x)
    {
        localSum += input[baseIdx + i];
    }
    shared[tid] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float meanVal = shared[0] / features;
    if (tid == 0) mean[rowIdx] = meanVal;
    __syncthreads();

    // Step 2: Compute variance
    float localVarSum = 0.0f;
    for (int i = tid; i < features; i += blockDim.x)
    {
        float diff = input[baseIdx + i] - meanVal;
        localVarSum += diff * diff;
    }
    shared[tid] = localVarSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float varVal = shared[0] / features;
    if (tid == 0) variance[rowIdx] = varVal;
    __syncthreads();

    // Step 3: Normalize and apply scale/shift
    float stdInv = rsqrtf(varVal + epsilon);
    for (int i = tid; i < features; i += blockDim.x)
    {
        float norm = (input[baseIdx + i] - meanVal) * stdInv;
        normalized[baseIdx + i] = norm;
        output[baseIdx + i] = gamma[i] * norm + beta[i];
    }
}

// Backward pass for layer norm
// Computes gradients for input, gamma, and beta
extern "C" __global__ void LayerNormBackward(
    const float* gradOutput,   // [batch, features]
    const float* input,        // [batch, features]
    const float* gamma,        // [features]
    const float* mean,         // [batch]
    const float* variance,     // [batch]
    const float* normalized,   // [batch, features]
    float* gradInput,          // [batch, features]
    float* gradGamma,          // [features] - accumulated across batch
    float* gradBeta,           // [features] - accumulated across batch
    int batch,
    int features,
    float epsilon)
{
    extern __shared__ float shared[];
    float* dNormShared = shared;
    float* dVarShared = shared + blockDim.x;
    float* dMeanShared = shared + 2 * blockDim.x;

    int batchIdx = blockIdx.x;
    int tid = threadIdx.x;

    if (batchIdx >= batch) return;

    int baseIdx = batchIdx * features;
    float meanVal = mean[batchIdx];
    float varVal = variance[batchIdx];
    float stdInv = rsqrtf(varVal + epsilon);

    // Step 1: Compute dNorm = gradOutput * gamma, and accumulate gradGamma, gradBeta
    float localDVarSum = 0.0f;
    float localDMeanSum = 0.0f;

    for (int i = tid; i < features; i += blockDim.x)
    {
        float grad = gradOutput[baseIdx + i];
        float norm = normalized[baseIdx + i];

        // Gradient for gamma and beta (need atomic add for multi-batch)
        atomicAdd(&gradGamma[i], grad * norm);
        atomicAdd(&gradBeta[i], grad);

        // Gradient w.r.t. normalized
        float dNorm = grad * gamma[i];

        // Gradient w.r.t. variance
        float xMinusMean = input[baseIdx + i] - meanVal;
        localDVarSum += dNorm * xMinusMean * -0.5f * powf(varVal + epsilon, -1.5f);

        // Gradient w.r.t. mean (partial)
        localDMeanSum += dNorm * -stdInv;
    }

    dVarShared[tid] = localDVarSum;
    dMeanShared[tid] = localDMeanSum;
    __syncthreads();

    // Reduce dVar
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            dVarShared[tid] += dVarShared[tid + stride];
        }
        __syncthreads();
    }
    float dVar = dVarShared[0];

    // Reduce dMean (partial)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            dMeanShared[tid] += dMeanShared[tid + stride];
        }
        __syncthreads();
    }

    // Complete dMean calculation
    float sumDiff = 0.0f;
    for (int i = tid; i < features; i += blockDim.x)
    {
        sumDiff += input[baseIdx + i] - meanVal;
    }
    dNormShared[tid] = sumDiff;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            dNormShared[tid] += dNormShared[tid + stride];
        }
        __syncthreads();
    }
    float dMean = dMeanShared[0] + dVar * -2.0f * dNormShared[0] / features;
    __syncthreads();

    // Step 2: Compute gradient w.r.t. input
    for (int i = tid; i < features; i += blockDim.x)
    {
        float grad = gradOutput[baseIdx + i];
        float dNorm = grad * gamma[i];

        gradInput[baseIdx + i] = dNorm * stdInv +
                                  dVar * 2.0f * (input[baseIdx + i] - meanVal) / features +
                                  dMean / features;
    }
}

// Backward pass for 3D tensors
extern "C" __global__ void LayerNormBackward3D(
    const float* gradOutput,   // [totalRows, features]
    const float* input,        // [totalRows, features]
    const float* gamma,        // [features]
    const float* mean,         // [totalRows]
    const float* variance,     // [totalRows]
    const float* normalized,   // [totalRows, features]
    float* gradInput,          // [totalRows, features]
    float* gradGamma,          // [features]
    float* gradBeta,           // [features]
    int totalRows,
    int features,
    float epsilon)
{
    extern __shared__ float shared[];
    float* dVarShared = shared;
    float* dMeanShared = shared + blockDim.x;
    float* dNormShared = shared + 2 * blockDim.x;

    int rowIdx = blockIdx.x;
    int tid = threadIdx.x;

    if (rowIdx >= totalRows) return;

    int baseIdx = rowIdx * features;
    float meanVal = mean[rowIdx];
    float varVal = variance[rowIdx];
    float stdInv = rsqrtf(varVal + epsilon);

    // Compute dNorm and accumulate gradGamma, gradBeta
    float localDVarSum = 0.0f;
    float localDMeanSum = 0.0f;

    for (int i = tid; i < features; i += blockDim.x)
    {
        float grad = gradOutput[baseIdx + i];
        float norm = normalized[baseIdx + i];

        atomicAdd(&gradGamma[i], grad * norm);
        atomicAdd(&gradBeta[i], grad);

        float dNorm = grad * gamma[i];
        float xMinusMean = input[baseIdx + i] - meanVal;
        localDVarSum += dNorm * xMinusMean * -0.5f * powf(varVal + epsilon, -1.5f);
        localDMeanSum += dNorm * -stdInv;
    }

    dVarShared[tid] = localDVarSum;
    dMeanShared[tid] = localDMeanSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            dVarShared[tid] += dVarShared[tid + stride];
            dMeanShared[tid] += dMeanShared[tid + stride];
        }
        __syncthreads();
    }
    float dVar = dVarShared[0];

    float sumDiff = 0.0f;
    for (int i = tid; i < features; i += blockDim.x)
    {
        sumDiff += input[baseIdx + i] - meanVal;
    }
    dNormShared[tid] = sumDiff;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            dNormShared[tid] += dNormShared[tid + stride];
        }
        __syncthreads();
    }
    float dMean = dMeanShared[0] + dVar * -2.0f * dNormShared[0] / features;
    __syncthreads();

    for (int i = tid; i < features; i += blockDim.x)
    {
        float grad = gradOutput[baseIdx + i];
        float dNorm = grad * gamma[i];

        gradInput[baseIdx + i] = dNorm * stdInv +
                                  dVar * 2.0f * (input[baseIdx + i] - meanVal) / features +
                                  dMean / features;
    }
}

// RMSNorm forward (alternative to LayerNorm, used in some Transformers)
// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
extern "C" __global__ void RMSNormForward(
    const float* input,      // [batch, features]
    const float* gamma,      // [features]
    float* output,           // [batch, features]
    float* rms,              // [batch] - cached for backward
    int batch,
    int features,
    float epsilon)
{
    extern __shared__ float shared[];

    int batchIdx = blockIdx.x;
    int tid = threadIdx.x;

    if (batchIdx >= batch) return;

    int baseIdx = batchIdx * features;

    // Compute mean of squares
    float localSumSq = 0.0f;
    for (int i = tid; i < features; i += blockDim.x)
    {
        float val = input[baseIdx + i];
        localSumSq += val * val;
    }
    shared[tid] = localSumSq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float rmsVal = sqrtf(shared[0] / features + epsilon);
    if (tid == 0) rms[batchIdx] = rmsVal;
    __syncthreads();

    // Normalize and scale
    float invRms = 1.0f / rmsVal;
    for (int i = tid; i < features; i += blockDim.x)
    {
        output[baseIdx + i] = input[baseIdx + i] * invRms * gamma[i];
    }
}
