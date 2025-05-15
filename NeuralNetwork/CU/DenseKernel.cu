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