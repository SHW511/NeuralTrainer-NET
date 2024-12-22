extern "C" __global__ void EmbeddingLookup(float* embeddings, float* inputs, float* output, int samples, int sequenceLength, int outputDim)
{
    int sample = blockIdx.x;
    int seq = threadIdx.x;

    if (sample < samples && seq < sequenceLength)
    {
        int index = (int)inputs[sample * sequenceLength + seq];
        for (int k = 0; k < outputDim; k++)
        {
            output[sample * sequenceLength * outputDim + seq * outputDim + k] = embeddings[index * outputDim + k];
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