extern "C" __global__ void EmbeddingLookup(float* embeddings, float* inputs, float* output, int samples, int sequenceLength, int outputDim)
{
    int sample = blockIdx.x;
    int seq = threadIdx.x;

    if (sample < samples && seq < sequenceLength)
    {
        int index = (int)inputs[sample * sequenceLength + seq];

        // Ensure index is within bounds
        if (index < 0 || index >= samples)
        {
            return;
        }

        for (int k = 0; k < outputDim; k++)
        {
            output[sample * sequenceLength * outputDim + seq * outputDim + k] = embeddings[index * outputDim + k];
        }
    }
}

// Embedding lookup with integer token indices (for Transformer models)
// tokens: [totalTokens] - flat array of token indices
// embeddings: [vocabSize, embDim] - embedding matrix
// output: [totalTokens, embDim] - output embeddings
// vocabSize: size of vocabulary for bounds checking
extern "C" __global__ void EmbeddingLookupInt(int* tokens, float* embeddings, float* output, int totalTokens, int embDim, int vocabSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalTokens)
    {
        int tokenId = tokens[idx];

        // Bounds check to prevent invalid memory access
        if (tokenId < 0 || tokenId >= vocabSize)
        {
            // Zero out the output for invalid tokens
            int outputOffset = idx * embDim;
            for (int d = 0; d < embDim; d++)
            {
                output[outputOffset + d] = 0.0f;
            }
            return;
        }

        int outputOffset = idx * embDim;
        int embOffset = tokenId * embDim;

        for (int d = 0; d < embDim; d++)
        {
            output[outputOffset + d] = embeddings[embOffset + d];
        }
    }
}

extern "C" __global__ void EmbeddingBackward(float* embeddings, float* gradient, int* inputIndices, int samples, int sequenceLength, int inputDim, int outputDim, float learningRate)
{
    int sample = blockIdx.x;
    int seq = threadIdx.x;

    if (sample < samples && seq < sequenceLength)
    {
        int index = inputIndices[sample * sequenceLength + seq];
        for (int k = 0; k < outputDim; k++)
        {
            atomicAdd(&embeddings[index * outputDim + k], -learningRate * gradient[sample * sequenceLength * outputDim + seq * outputDim + k]);
        }
    }
}