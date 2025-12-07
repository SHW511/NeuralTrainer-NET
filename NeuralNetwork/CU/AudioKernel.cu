// AudioKernel.cu - CUDA kernels for audio processing

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void ApplyWindow(
    const float* input,
    const float* window,
    float* output,
    int frameSize,
    int hopLength,
    int numFrames)
{
    int frame = blockIdx.x;
    int i = threadIdx.x;

    if (frame >= numFrames || i >= frameSize) return;

    int inputIdx = frame * hopLength + i;
    output[frame * frameSize + i] = input[inputIdx] * window[i];
}

extern "C" __global__ void ComputeMagnitudePhase(
    const float2* fftOutput,  // Complex output from cuFFT
    float* magnitude,
    float* phase,
    int numBins,
    int numFrames)
{
    int frame = blockIdx.x;
    int bin = threadIdx.x;

    if (frame >= numFrames || bin >= numBins) return;

    int idx = frame * numBins + bin;
    float real = fftOutput[idx].x;
    float imag = fftOutput[idx].y;

    magnitude[idx] = sqrtf(real * real + imag * imag);
    phase[idx] = atan2f(imag, real);
}

extern "C" __global__ void ApplyMelFilterbank(
    const float* magnitude,   // [numFrames, numBins]
    const float* filterbank,  // [melBins, numBins]
    float* melSpec,           // [numFrames, melBins]
    int numFrames,
    int numBins,
    int melBins)
{
    int frame = blockIdx.x;
    int mel = threadIdx.x;

    if (frame >= numFrames || mel >= melBins) return;

    float sum = 0.0f;
    for (int f = 0; f < numBins; f++)
    {
        sum += magnitude[frame * numBins + f] * filterbank[mel * numBins + f];
    }
    melSpec[frame * melBins + mel] = sum;
}

extern "C" __global__ void InvertMelFilterbank(
    const float* melSpec,     // [numFrames, melBins]
    const float* filterbank,  // [melBins, numBins]
    const float* filterSums,  // [numBins]
    float* magnitude,         // [numFrames, numBins]
    int numFrames,
    int numBins,
    int melBins)
{
    int frame = blockIdx.x;
    int bin = threadIdx.x;

    if (frame >= numFrames || bin >= numBins) return;

    float sum = 0.0f;
    for (int m = 0; m < melBins; m++)
    {
        sum += melSpec[frame * melBins + m] * filterbank[m * numBins + bin];
    }
    magnitude[frame * numBins + bin] = sum / fmaxf(filterSums[bin], 1e-10f);
}

extern "C" __global__ void ReconstructComplex(
    const float* magnitude,
    const float* phase,
    float2* complex,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float mag = magnitude[idx];
    float ph = phase[idx];
    complex[idx].x = mag * cosf(ph);
    complex[idx].y = mag * sinf(ph);
}

extern "C" __global__ void OverlapAdd(
    const float* frames,      // [numFrames, frameSize]
    const float* window,
    float* output,
    float* windowSum,
    int frameSize,
    int hopLength,
    int numFrames,
    int outputLength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outputLength) return;

    float sum = 0.0f;
    float winSum = 0.0f;

    for (int frame = 0; frame < numFrames; frame++)
    {
        int frameStart = frame * hopLength;
        int posInFrame = idx - frameStart;

        if (posInFrame >= 0 && posInFrame < frameSize)
        {
            float win = window[posInFrame];
            sum += frames[frame * frameSize + posInFrame] * win;
            winSum += win * win;
        }
    }

    output[idx] = sum;
    windowSum[idx] = winSum;
}

extern "C" __global__ void DbToLinear(
    const float* dbSpec,
    float* linearSpec,
    float refLevel,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    linearSpec[idx] = powf(10.0f, dbSpec[idx] / refLevel);
}

extern "C" __global__ void LinearToDb(
    const float* linearSpec,
    float* dbSpec,
    float refLevel,
    float minLevel,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float val = fmaxf(linearSpec[idx], 1e-10f);
    val = refLevel * log10f(val);
    dbSpec[idx] = fmaxf(val, minLevel);
}
