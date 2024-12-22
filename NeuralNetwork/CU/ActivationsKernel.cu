#include <cfloat>
extern "C" __global__ void ReLU(float* inputs, float* outputs, int rows, int cols, float max_value, float threshold, float negative_slope)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols)
    {
        float x = inputs[i * cols + j];
        if (x >= max_value)
        {
            outputs[i * cols + j] = max_value;
        }
        else if (x >= threshold)
        {
            outputs[i * cols + j] = x;
        }
        else
        {
            outputs[i * cols + j] = negative_slope * (x - threshold);
        }
    }
}

extern "C" __global__ void Sigmoid(float* inputs, float* outputs, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols)
    {
        float x = inputs[i * cols + j];
        outputs[i * cols + j] = 1.0f / (1.0f + exp(-x));
    }
}

extern "C" __global__ void SoftMax(float* inputs, float* outputs, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols)
    {
        // Find the maximum value in the row
        float max = -FLT_MAX;
        for (int k = 0; k < cols; k++)
        {
            max = fmaxf(max, inputs[i * cols + k]);
        }

        // Compute the exponentials and sum them up
        float sum = 0.0f;
        for (int k = 0; k < cols; k++)
        {
            outputs[i * cols + k] = exp(inputs[i * cols + k] - max);
            sum += outputs[i * cols + k];
        }

        // Normalize the values to get the SoftMax output
        for (int k = 0; k < cols; k++)
        {
            outputs[i * cols + k] /= sum;
        }
    }
}