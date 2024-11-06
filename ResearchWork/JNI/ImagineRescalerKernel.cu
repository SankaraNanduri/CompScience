#include <cuda_runtime.h>

// CUDA kernel to rescale pixel values
__global__ void rescalePixels(short* inputImage, unsigned char* outputImage, int innerBound, int upperBound, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If we're within range, rescale the pixel
    if (idx < size) {
        short pixel = inputImage[idx];
        float normalized = (float)(pixel - innerBound) / (upperBound - innerBound);
        outputImage[idx] = (unsigned char)(normalized * 255.0f); // Scale to 0-255 range
    }
}
