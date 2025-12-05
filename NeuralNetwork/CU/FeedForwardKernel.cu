// FeedForwardKernel.cu - CUDA kernels for Feed-Forward Network operations
// FFN(x) = Linear2(GELU(Linear1(x)))

// GELU activation function: GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Using fast approximation from BERT paper
extern "C" __global__ void GELU(
    const float* input,
    float* output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float x = input[idx];
        // Fast GELU approximation
        float c = 0.7978845608f;  // sqrt(2/pi)
        float inner = c * (x + 0.044715f * x * x * x);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// GELU backward: d/dx GELU(x)
// Using derivative of approximation
extern "C" __global__ void GELUBackward(
    const float* input,       // Original input (before GELU)
    const float* gradOutput,  // Gradient from next layer
    float* gradInput,         // Gradient to propagate back
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float x = input[idx];
        float c = 0.7978845608f;  // sqrt(2/pi)

        // Compute tanh(inner) and sech^2(inner)
        float inner = c * (x + 0.044715f * x * x * x);
        float tanh_inner = tanhf(inner);

        // d/dx tanh(inner) = sech^2(inner) * d(inner)/dx
        float sech2 = 1.0f - tanh_inner * tanh_inner;
        float d_inner = c * (1.0f + 3.0f * 0.044715f * x * x);

        // d/dx GELU(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner
        float d_gelu = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner;

        gradInput[idx] = gradOutput[idx] * d_gelu;
    }
}

// Fused Linear + GELU: output = GELU(input @ weights + bias)
// More efficient than separate operations due to reduced memory traffic
extern "C" __global__ void LinearGELU(
    const float* input,    // [batch, inputDim]
    const float* weights,  // [inputDim, outputDim]
    const float* bias,     // [outputDim]
    float* output,         // [batch, outputDim]
    float* preGelu,        // [batch, outputDim] - cached for backward
    int batch,
    int inputDim,
    int outputDim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch && col < outputDim)
    {
        float sum = bias[col];
        for (int k = 0; k < inputDim; k++)
        {
            sum += input[row * inputDim + k] * weights[k * outputDim + col];
        }

        // Cache pre-GELU value for backward
        preGelu[row * outputDim + col] = sum;

        // Apply GELU
        float c = 0.7978845608f;
        float inner = c * (sum + 0.044715f * sum * sum * sum);
        output[row * outputDim + col] = 0.5f * sum * (1.0f + tanhf(inner));
    }
}

// Linear layer forward: output = input @ weights + bias
extern "C" __global__ void LinearForward(
    const float* input,    // [batch, inputDim]
    const float* weights,  // [inputDim, outputDim]
    const float* bias,     // [outputDim]
    float* output,         // [batch, outputDim]
    int batch,
    int inputDim,
    int outputDim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch && col < outputDim)
    {
        float sum = bias[col];
        for (int k = 0; k < inputDim; k++)
        {
            sum += input[row * inputDim + k] * weights[k * outputDim + col];
        }
        output[row * outputDim + col] = sum;
    }
}

// Linear layer backward: compute gradients for weights, bias, and input
extern "C" __global__ void LinearBackward(
    const float* gradOutput,  // [batch, outputDim]
    const float* input,       // [batch, inputDim]
    const float* weights,     // [inputDim, outputDim]
    float* gradInput,         // [batch, inputDim]
    float* gradWeights,       // [inputDim, outputDim]
    float* gradBias,          // [outputDim]
    int batch,
    int inputDim,
    int outputDim)
{
    // This kernel handles weight and bias gradients
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // inputDim
    int o = blockIdx.x * blockDim.x + threadIdx.x;  // outputDim

    if (i < inputDim && o < outputDim)
    {
        float wGrad = 0.0f;
        for (int b = 0; b < batch; b++)
        {
            wGrad += input[b * inputDim + i] * gradOutput[b * outputDim + o];
        }
        atomicAdd(&gradWeights[i * outputDim + o], wGrad);
    }

    // Bias gradient (first row of threads)
    if (i == 0 && o < outputDim)
    {
        float bGrad = 0.0f;
        for (int b = 0; b < batch; b++)
        {
            bGrad += gradOutput[b * outputDim + o];
        }
        atomicAdd(&gradBias[o], bGrad);
    }
}

// Compute gradient w.r.t. input in linear layer
extern "C" __global__ void LinearBackwardInput(
    const float* gradOutput,  // [batch, outputDim]
    const float* weights,     // [inputDim, outputDim]
    float* gradInput,         // [batch, inputDim]
    int batch,
    int inputDim,
    int outputDim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // inputDim

    if (row < batch && col < inputDim)
    {
        float sum = 0.0f;
        for (int o = 0; o < outputDim; o++)
        {
            sum += gradOutput[row * outputDim + o] * weights[col * outputDim + o];
        }
        gradInput[row * inputDim + col] = sum;
    }
}

// Dropout forward pass
extern "C" __global__ void Dropout(
    const float* input,
    const float* mask,    // Random values in [0,1)
    float* output,
    float dropoutRate,
    float scale,
    int size,
    bool training)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        if (training)
        {
            // Apply dropout: set to 0 if mask < dropoutRate, otherwise scale
            output[idx] = (mask[idx] >= dropoutRate) ? input[idx] * scale : 0.0f;
        }
        else
        {
            // During inference, no dropout
            output[idx] = input[idx];
        }
    }
}

// Residual connection: output = input1 + input2
extern "C" __global__ void ResidualAdd(
    const float* input1,
    const float* input2,
    float* output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = input1[idx] + input2[idx];
    }
}

// Fused FFN: Linear1 + GELU + Linear2
// This is the most efficient version but requires more shared memory
extern "C" __global__ void FusedFFN(
    const float* input,     // [batch, modelDim]
    const float* w1,        // [modelDim, ffDim]
    const float* b1,        // [ffDim]
    const float* w2,        // [ffDim, modelDim]
    const float* b2,        // [modelDim]
    float* output,          // [batch, modelDim]
    float* hidden,          // [batch, ffDim] - cached for backward
    float* preGelu,         // [batch, ffDim] - cached for backward
    int batch,
    int modelDim,
    int ffDim)
{
    // Each block handles one row (one position)
    extern __shared__ float shared[];

    int batchIdx = blockIdx.x;
    int tid = threadIdx.x;

    if (batchIdx >= batch) return;

    int inputOffset = batchIdx * modelDim;
    int hiddenOffset = batchIdx * ffDim;

    // Step 1: Compute hidden = GELU(input @ w1 + b1)
    // Each thread computes some outputs in hidden dimension
    for (int ff = tid; ff < ffDim; ff += blockDim.x)
    {
        float sum = b1[ff];
        for (int d = 0; d < modelDim; d++)
        {
            sum += input[inputOffset + d] * w1[d * ffDim + ff];
        }

        // Cache pre-GELU
        preGelu[hiddenOffset + ff] = sum;

        // Apply GELU
        float c = 0.7978845608f;
        float inner = c * (sum + 0.044715f * sum * sum * sum);
        float h = 0.5f * sum * (1.0f + tanhf(inner));
        hidden[hiddenOffset + ff] = h;
    }
    __syncthreads();

    // Step 2: Compute output = hidden @ w2 + b2
    for (int d = tid; d < modelDim; d += blockDim.x)
    {
        float sum = b2[d];
        for (int ff = 0; ff < ffDim; ff++)
        {
            sum += hidden[hiddenOffset + ff] * w2[ff * modelDim + d];
        }
        output[inputOffset + d] = sum;
    }
}

// SiLU/Swish activation: SiLU(x) = x * sigmoid(x)
// Alternative to GELU used in some architectures like LLaMA
extern "C" __global__ void SiLU(
    const float* input,
    float* output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}

// SiLU backward
extern "C" __global__ void SiLUBackward(
    const float* input,
    const float* gradOutput,
    float* gradInput,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        // d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //              = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        float d_silu = sigmoid * (1.0f + x * (1.0f - sigmoid));
        gradInput[idx] = gradOutput[idx] * d_silu;
    }
}
