// TTSKernel.cu - CUDA kernels for Text-to-Speech operations
// Optimized for RTX 5090 (Blackwell, Compute Capability 12.0)

#include <cuda_runtime.h>

// =============================================================================
// Conv1D Forward - For encoder and postnet convolutions
// =============================================================================

// 1D Convolution: input[seqLen, inChannels] * weights[kernelSize*inChannels, outChannels] + bias
// Output: [seqLen, outChannels]
extern "C" __global__ void Conv1DForward(
    const float* input,      // [seqLen, inChannels]
    const float* weights,    // [kernelSize * inChannels, outChannels]
    const float* bias,       // [outChannels]
    float* output,           // [seqLen, outChannels]
    int seqLen,
    int inChannels,
    int outChannels,
    int kernelSize)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;  // Time position
    int oc = blockIdx.y * blockDim.y + threadIdx.y; // Output channel

    if (t >= seqLen || oc >= outChannels) return;

    int padding = kernelSize / 2;
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

    output[t * outChannels + oc] = sum;
}

// Conv1D with ReLU activation fused
extern "C" __global__ void Conv1DForwardReLU(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int seqLen,
    int inChannels,
    int outChannels,
    int kernelSize)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;

    if (t >= seqLen || oc >= outChannels) return;

    int padding = kernelSize / 2;
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

    // ReLU
    output[t * outChannels + oc] = fmaxf(0.0f, sum);
}

// Conv1D with Tanh activation fused
extern "C" __global__ void Conv1DForwardTanh(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int seqLen,
    int inChannels,
    int outChannels,
    int kernelSize)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;

    if (t >= seqLen || oc >= outChannels) return;

    int padding = kernelSize / 2;
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

    // Tanh
    output[t * outChannels + oc] = tanhf(sum);
}

// =============================================================================
// Embedding Lookup
// =============================================================================

extern "C" __global__ void EmbeddingLookup(
    const int* tokens,           // [seqLen]
    const float* embeddings,     // [vocabSize, embedDim]
    float* output,               // [seqLen, embedDim]
    int seqLen,
    int embedDim)
{
    int t = blockIdx.x;          // Token position
    int d = threadIdx.x;         // Embedding dimension

    if (t >= seqLen || d >= embedDim) return;

    int tokenId = tokens[t];
    output[t * embedDim + d] = embeddings[tokenId * embedDim + d];
}

// =============================================================================
// Prenet Forward - Dense layers with ReLU and dropout
// =============================================================================

extern "C" __global__ void PrenetForward(
    const float* input,          // [inputDim]
    const float* weights,        // [inputDim, outputDim]
    const float* bias,           // [outputDim]
    float* output,               // [outputDim]
    int inputDim,
    int outputDim)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;

    if (o >= outputDim) return;

    float sum = bias[o];
    for (int i = 0; i < inputDim; i++)
    {
        sum += input[i] * weights[i * outputDim + o];
    }

    // ReLU
    output[o] = fmaxf(0.0f, sum);
}

// =============================================================================
// Location-Sensitive Attention for TTS
// =============================================================================

// Compute attention energies with location features
extern "C" __global__ void LocationSensitiveAttentionEnergy(
    const float* query,              // [attentionDim] - projected decoder state
    const float* keys,               // [encoderLen, attentionDim] - projected encoder outputs
    const float* prevAttnWeights,    // [encoderLen] - cumulative attention
    const float* locationConv,       // [kernelSize, numFilters]
    const float* locationBias,       // [numFilters]
    const float* attentionV,         // [attentionDim + numFilters]
    float* energies,                 // [encoderLen]
    int encoderLen,
    int attentionDim,
    int numFilters,
    int locationKernelSize)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (e >= encoderLen) return;

    // Compute location features via 1D convolution
    float locationFeatures[64];  // Assuming max numFilters = 64
    int padding = locationKernelSize / 2;

    for (int f = 0; f < numFilters; f++)
    {
        float sum = locationBias[f];
        for (int k = 0; k < locationKernelSize; k++)
        {
            int idx = e + k - padding;
            if (idx >= 0 && idx < encoderLen)
            {
                sum += prevAttnWeights[idx] * locationConv[k * numFilters + f];
            }
        }
        locationFeatures[f] = sum;
    }

    // Compute energy: V * tanh(query + key + location_features)
    float energy = 0.0f;
    for (int d = 0; d < attentionDim; d++)
    {
        float combined = tanhf(query[d] + keys[e * attentionDim + d]);
        energy += combined * attentionV[d];
    }
    for (int f = 0; f < numFilters; f++)
    {
        energy += locationFeatures[f] * attentionV[attentionDim + f];
    }

    energies[e] = energy;
}

// Softmax for attention weights
extern "C" __global__ void AttentionSoftmax(
    const float* energies,   // [encoderLen]
    float* weights,          // [encoderLen]
    int encoderLen)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    // Find max
    float localMax = -1e30f;
    for (int i = tid; i < encoderLen; i += blockDim.x)
    {
        if (energies[i] > localMax) localMax = energies[i];
    }
    shared[tid] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride && shared[tid + stride] > shared[tid])
            shared[tid] = shared[tid + stride];
        __syncthreads();
    }
    float maxVal = shared[0];
    __syncthreads();

    // Exp and sum
    float localSum = 0.0f;
    for (int i = tid; i < encoderLen; i += blockDim.x)
    {
        float expVal = expf(energies[i] - maxVal);
        weights[i] = expVal;
        localSum += expVal;
    }
    shared[tid] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    float sumVal = shared[0];
    __syncthreads();

    // Normalize
    float invSum = 1.0f / sumVal;
    for (int i = tid; i < encoderLen; i += blockDim.x)
    {
        weights[i] *= invSum;
    }
}

// Compute context vector from attention weights
extern "C" __global__ void AttentionContext(
    const float* attnWeights,    // [encoderLen]
    const float* encoderOutput,  // [encoderLen, encoderDim]
    float* context,              // [encoderDim]
    int encoderLen,
    int encoderDim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (d >= encoderDim) return;

    float sum = 0.0f;
    for (int e = 0; e < encoderLen; e++)
    {
        sum += attnWeights[e] * encoderOutput[e * encoderDim + d];
    }
    context[d] = sum;
}

// =============================================================================
// LSTM Step for TTS Decoder
// =============================================================================

// Single LSTM step: computes new hidden and cell states
extern "C" __global__ void LSTMStep(
    const float* input,          // [inputDim]
    const float* prevH,          // [hiddenDim]
    const float* prevC,          // [hiddenDim]
    const float* weightsIh,      // [inputDim, hiddenDim * 4]
    const float* weightsHh,      // [hiddenDim, hiddenDim * 4]
    const float* bias,           // [hiddenDim * 4]
    float* newH,                 // [hiddenDim]
    float* newC,                 // [hiddenDim]
    int inputDim,
    int hiddenDim)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= hiddenDim) return;

    // Compute all 4 gates for this hidden unit
    float gates[4];
    for (int g = 0; g < 4; g++)
    {
        int gateIdx = g * hiddenDim + h;
        float sum = bias[gateIdx];

        // Input contribution
        for (int i = 0; i < inputDim; i++)
        {
            sum += input[i] * weightsIh[i * hiddenDim * 4 + gateIdx];
        }

        // Hidden contribution
        for (int hh = 0; hh < hiddenDim; hh++)
        {
            sum += prevH[hh] * weightsHh[hh * hiddenDim * 4 + gateIdx];
        }

        gates[g] = sum;
    }

    // Apply activations and compute output
    float i_t = 1.0f / (1.0f + expf(-gates[0]));  // Input gate (sigmoid)
    float f_t = 1.0f / (1.0f + expf(-gates[1]));  // Forget gate (sigmoid)
    float g_t = tanhf(gates[2]);                   // Cell gate (tanh)
    float o_t = 1.0f / (1.0f + expf(-gates[3]));  // Output gate (sigmoid)

    float c_new = f_t * prevC[h] + i_t * g_t;
    float h_new = o_t * tanhf(c_new);

    newC[h] = c_new;
    newH[h] = h_new;
}

// Optimized LSTM step using shared memory
extern "C" __global__ void LSTMStepOptimized(
    const float* input,          // [inputDim]
    const float* prevH,          // [hiddenDim]
    const float* prevC,          // [hiddenDim]
    const float* weightsIh,      // [inputDim, hiddenDim * 4]
    const float* weightsHh,      // [hiddenDim, hiddenDim * 4]
    const float* bias,           // [hiddenDim * 4]
    float* newH,                 // [hiddenDim]
    float* newC,                 // [hiddenDim]
    int inputDim,
    int hiddenDim)
{
    extern __shared__ float shared[];
    float* sharedInput = shared;
    float* sharedPrevH = shared + inputDim;

    int tid = threadIdx.x;
    int h = blockIdx.x * blockDim.x + tid;

    // Load input and prevH to shared memory
    if (tid < inputDim) sharedInput[tid] = input[tid];
    if (tid < hiddenDim) sharedPrevH[tid] = prevH[tid];
    __syncthreads();

    if (h >= hiddenDim) return;

    // Compute gates
    float gate_i = bias[h];
    float gate_f = bias[hiddenDim + h];
    float gate_g = bias[2 * hiddenDim + h];
    float gate_o = bias[3 * hiddenDim + h];

    for (int i = 0; i < inputDim; i++)
    {
        float inp = sharedInput[i];
        gate_i += inp * weightsIh[i * hiddenDim * 4 + h];
        gate_f += inp * weightsIh[i * hiddenDim * 4 + hiddenDim + h];
        gate_g += inp * weightsIh[i * hiddenDim * 4 + 2 * hiddenDim + h];
        gate_o += inp * weightsIh[i * hiddenDim * 4 + 3 * hiddenDim + h];
    }

    for (int hh = 0; hh < hiddenDim; hh++)
    {
        float ph = sharedPrevH[hh];
        gate_i += ph * weightsHh[hh * hiddenDim * 4 + h];
        gate_f += ph * weightsHh[hh * hiddenDim * 4 + hiddenDim + h];
        gate_g += ph * weightsHh[hh * hiddenDim * 4 + 2 * hiddenDim + h];
        gate_o += ph * weightsHh[hh * hiddenDim * 4 + 3 * hiddenDim + h];
    }

    // Activations
    float i_t = 1.0f / (1.0f + expf(-gate_i));
    float f_t = 1.0f / (1.0f + expf(-gate_f));
    float g_t = tanhf(gate_g);
    float o_t = 1.0f / (1.0f + expf(-gate_o));

    float c_new = f_t * prevC[h] + i_t * g_t;
    float h_new = o_t * tanhf(c_new);

    newC[h] = c_new;
    newH[h] = h_new;
}

// =============================================================================
// Mel Projection - Project decoder output to mel spectrogram
// =============================================================================

extern "C" __global__ void MelProjection(
    const float* input,          // [inputDim]
    const float* weights,        // [inputDim, outputDim]
    const float* bias,           // [outputDim]
    float* output,               // [outputDim]
    int inputDim,
    int outputDim)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;

    if (o >= outputDim) return;

    float sum = bias[o];
    for (int i = 0; i < inputDim; i++)
    {
        sum += input[i] * weights[i * outputDim + o];
    }
    output[o] = sum;
}

// =============================================================================
// Postnet residual addition
// =============================================================================

extern "C" __global__ void PostnetResidual(
    const float* melOutput,      // [frames, melBins]
    const float* postnetOutput,  // [frames, melBins]
    float* result,               // [frames, melBins]
    int totalSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= totalSize) return;

    result[idx] = melOutput[idx] + postnetOutput[idx];
}

// =============================================================================
// Stop Token Prediction
// =============================================================================

extern "C" __global__ void StopTokenSigmoid(
    const float* logit,          // [1]
    float* probability,          // [1]
    int dummy)  // Unused, for consistent kernel signature
{
    if (threadIdx.x == 0)
    {
        probability[0] = 1.0f / (1.0f + expf(-logit[0]));
    }
}

// =============================================================================
// Batch operations for training efficiency
// =============================================================================

// Batched embedding lookup
extern "C" __global__ void BatchedEmbeddingLookup(
    const int* tokens,           // [batchSize, seqLen]
    const float* embeddings,     // [vocabSize, embedDim]
    float* output,               // [batchSize, seqLen, embedDim]
    int batchSize,
    int seqLen,
    int embedDim)
{
    int b = blockIdx.z;          // Batch index
    int t = blockIdx.x;          // Token position
    int d = threadIdx.x;         // Embedding dimension

    if (b >= batchSize || t >= seqLen || d >= embedDim) return;

    int tokenId = tokens[b * seqLen + t];
    output[(b * seqLen + t) * embedDim + d] = embeddings[tokenId * embedDim + d];
}

// Batched Conv1D for parallel processing
extern "C" __global__ void BatchedConv1DForward(
    const float* input,          // [batchSize, seqLen, inChannels]
    const float* weights,        // [kernelSize * inChannels, outChannels]
    const float* bias,           // [outChannels]
    float* output,               // [batchSize, seqLen, outChannels]
    int batchSize,
    int seqLen,
    int inChannels,
    int outChannels,
    int kernelSize)
{
    int b = blockIdx.z;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batchSize || t >= seqLen || oc >= outChannels) return;

    int padding = kernelSize / 2;
    float sum = bias[oc];

    for (int k = 0; k < kernelSize; k++)
    {
        int inputIdx = t + k - padding;
        if (inputIdx >= 0 && inputIdx < seqLen)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                int weightIdx = (k * inChannels + ic) * outChannels + oc;
                int inIdx = (b * seqLen + inputIdx) * inChannels + ic;
                sum += input[inIdx] * weights[weightIdx];
            }
        }
    }

    output[(b * seqLen + t) * outChannels + oc] = sum;
}

// =============================================================================
// Gradient computation kernels
// =============================================================================

// Conv1D backward for weights
extern "C" __global__ void Conv1DBackwardWeights(
    const float* input,          // [seqLen, inChannels]
    const float* gradOutput,     // [seqLen, outChannels]
    float* gradWeights,          // [kernelSize * inChannels, outChannels]
    int seqLen,
    int inChannels,
    int outChannels,
    int kernelSize)
{
    int wIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWeights = kernelSize * inChannels * outChannels;

    if (wIdx >= totalWeights) return;

    int oc = wIdx % outChannels;
    int icKernel = wIdx / outChannels;
    int k = icKernel / inChannels;
    int ic = icKernel % inChannels;

    int padding = kernelSize / 2;
    float sum = 0.0f;

    for (int t = 0; t < seqLen; t++)
    {
        int inputIdx = t + k - padding;
        if (inputIdx >= 0 && inputIdx < seqLen)
        {
            sum += input[inputIdx * inChannels + ic] * gradOutput[t * outChannels + oc];
        }
    }

    gradWeights[wIdx] = sum;
}

// Conv1D backward for input
extern "C" __global__ void Conv1DBackwardInput(
    const float* gradOutput,     // [seqLen, outChannels]
    const float* weights,        // [kernelSize * inChannels, outChannels]
    float* gradInput,            // [seqLen, inChannels]
    int seqLen,
    int inChannels,
    int outChannels,
    int kernelSize)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int ic = blockIdx.y * blockDim.y + threadIdx.y;

    if (t >= seqLen || ic >= inChannels) return;

    int padding = kernelSize / 2;
    float sum = 0.0f;

    for (int k = 0; k < kernelSize; k++)
    {
        int outputIdx = t - k + padding;
        if (outputIdx >= 0 && outputIdx < seqLen)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                int weightIdx = (k * inChannels + ic) * outChannels + oc;
                sum += gradOutput[outputIdx * outChannels + oc] * weights[weightIdx];
            }
        }
    }

    gradInput[t * inChannels + ic] = sum;
}

// Embedding backward - accumulate gradients into embedding table
extern "C" __global__ void EmbeddingBackward(
    const int* tokens,           // [seqLen]
    const float* gradOutput,     // [seqLen, embedDim]
    float* gradEmbeddings,       // [vocabSize, embedDim] - accumulated
    int seqLen,
    int embedDim)
{
    int t = blockIdx.x;
    int d = threadIdx.x;
    if (t >= seqLen || d >= embedDim) return;

    int tokenId = tokens[t];
    atomicAdd(&gradEmbeddings[tokenId * embedDim + d], gradOutput[t * embedDim + d]);
}

// LSTM backward - backprop through LSTM cell
extern "C" __global__ void LSTMStepBackward(
    const float* gradH,          // [hiddenDim] - gradient from next layer
    const float* gradCNext,      // [hiddenDim] - gradient from next timestep
    const float* gates,          // [hiddenDim * 4] - saved from forward
    const float* prevC,          // [hiddenDim]
    const float* newC,           // [hiddenDim]
    const float* prevH,          // [hiddenDim]
    const float* input,          // [inputDim]
    const float* weightsIh,      // [inputDim, hiddenDim * 4]
    const float* weightsHh,      // [hiddenDim, hiddenDim * 4]
    float* gradInput,            // [inputDim]
    float* gradPrevH,            // [hiddenDim]
    float* gradPrevC,            // [hiddenDim]
    float* gradWeightsIh,        // [inputDim, hiddenDim * 4]
    float* gradWeightsHh,        // [hiddenDim, hiddenDim * 4]
    float* gradBias,             // [hiddenDim * 4]
    int inputDim,
    int hiddenDim)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hiddenDim) return;

    // Retrieve gate values (after sigmoid/tanh)
    float i_t = gates[h];
    float f_t = gates[hiddenDim + h];
    float g_t = gates[2 * hiddenDim + h];
    float o_t = gates[3 * hiddenDim + h];

    float c_new = newC[h];
    float tanh_c = tanhf(c_new);

    // Gradient of output
    float dh = gradH[h];
    float dc = gradCNext[h];

    // Backprop through output gate
    dc += dh * o_t * (1.0f - tanh_c * tanh_c);  // through tanh
    float do_t = dh * tanh_c;

    // Backprop through cell update
    float di_t = dc * g_t;
    float df_t = dc * prevC[h];
    float dg_t = dc * i_t;

    // Gradient to previous cell state
    gradPrevC[h] = dc * f_t;

    // Backprop through gate activations
    float di_raw = di_t * i_t * (1.0f - i_t);  // sigmoid derivative
    float df_raw = df_t * f_t * (1.0f - f_t);
    float dg_raw = dg_t * (1.0f - g_t * g_t);  // tanh derivative
    float do_raw = do_t * o_t * (1.0f - o_t);

    // Accumulate bias gradients
    atomicAdd(&gradBias[h], di_raw);
    atomicAdd(&gradBias[hiddenDim + h], df_raw);
    atomicAdd(&gradBias[2 * hiddenDim + h], dg_raw);
    atomicAdd(&gradBias[3 * hiddenDim + h], do_raw);

    // Compute gradient w.r.t. previous hidden state
    float dh_prev = 0.0f;
    for (int hh = 0; hh < hiddenDim; hh++)
    {
        float dgate_i = (hh == h) ? di_raw : 0;
        float dgate_f = (hh == h) ? df_raw : 0;
        float dgate_g = (hh == h) ? dg_raw : 0;
        float dgate_o = (hh == h) ? do_raw : 0;

        dh_prev += weightsHh[h * hiddenDim * 4 + hh] * dgate_i;
        dh_prev += weightsHh[h * hiddenDim * 4 + hiddenDim + hh] * dgate_f;
        dh_prev += weightsHh[h * hiddenDim * 4 + 2 * hiddenDim + hh] * dgate_g;
        dh_prev += weightsHh[h * hiddenDim * 4 + 3 * hiddenDim + hh] * dgate_o;
    }
    gradPrevH[h] = dh_prev;
}

// Prenet backward - backprop through prenet layer
extern "C" __global__ void PrenetBackward(
    const float* gradOutput,     // [outputDim]
    const float* output,         // [outputDim] - saved from forward (after ReLU)
    const float* input,          // [inputDim]
    const float* weights,        // [inputDim, outputDim]
    float* gradInput,            // [inputDim]
    float* gradWeights,          // [inputDim, outputDim]
    float* gradBias,             // [outputDim]
    int inputDim,
    int outputDim)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= outputDim) return;

    // ReLU backward
    float grad = (output[o] > 0) ? gradOutput[o] : 0;

    // Bias gradient
    atomicAdd(&gradBias[o], grad);

    // Weight gradients
    for (int i = 0; i < inputDim; i++)
    {
        atomicAdd(&gradWeights[i * outputDim + o], input[i] * grad);
    }
}

// Mel loss gradient - MSE derivative
extern "C" __global__ void MelLossGradient(
    const float* predicted,      // [frames, melBins]
    const float* target,         // [frames, melBins]
    float* gradient,             // [frames, melBins]
    int totalSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    // MSE gradient: 2 * (predicted - target) / N
    gradient[idx] = 2.0f * (predicted[idx] - target[idx]) / (float)totalSize;
}

// Attention backward - backprop through attention mechanism
extern "C" __global__ void AttentionBackward(
    const float* gradContext,    // [encoderDim]
    const float* attnWeights,    // [encoderLen]
    const float* encoderOutput,  // [encoderLen, encoderDim]
    float* gradAttnWeights,      // [encoderLen]
    float* gradEncoderOutput,    // [encoderLen, encoderDim]
    int encoderLen,
    int encoderDim)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= encoderLen) return;

    // Gradient w.r.t. attention weights
    float gradWeight = 0.0f;
    for (int d = 0; d < encoderDim; d++)
    {
        gradWeight += gradContext[d] * encoderOutput[e * encoderDim + d];
    }
    gradAttnWeights[e] = gradWeight;

    // Gradient w.r.t. encoder output
    for (int d = 0; d < encoderDim; d++)
    {
        atomicAdd(&gradEncoderOutput[e * encoderDim + d], gradContext[d] * attnWeights[e]);
    }
}

// Softmax backward - backprop through softmax activation
extern "C" __global__ void SoftmaxBackward(
    const float* gradOutput,     // [n]
    const float* softmaxOutput,  // [n]
    float* gradInput,            // [n]
    int n)
{
    extern __shared__ float shared[];
    int tid = threadIdx.x;

    // Compute dot product: sum(gradOutput * softmaxOutput)
    float localDot = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
    {
        localDot += gradOutput[i] * softmaxOutput[i];
    }
    shared[tid] = localDot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    float dotProduct = shared[0];
    __syncthreads();

    // Compute gradient: softmax * (gradOutput - dotProduct)
    for (int i = tid; i < n; i += blockDim.x)
    {
        gradInput[i] = softmaxOutput[i] * (gradOutput[i] - dotProduct);
    }
}

// Adam optimizer update
extern "C" __global__ void AdamUpdate(
    float* params,               // Parameters to update
    const float* gradients,      // Gradients
    float* m,                    // First moment
    float* v,                    // Second moment
    float lr,                    // Learning rate
    float beta1,                 // Beta1 (0.9)
    float beta2,                 // Beta2 (0.999)
    float epsilon,               // Epsilon (1e-8)
    float beta1_t,               // beta1^t
    float beta2_t,               // beta2^t
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = gradients[idx];

    // Update moments
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    // Bias correction
    float m_hat = m[idx] / (1.0f - beta1_t);
    float v_hat = v[idx] / (1.0f - beta2_t);

    // Update parameters
    params[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}
