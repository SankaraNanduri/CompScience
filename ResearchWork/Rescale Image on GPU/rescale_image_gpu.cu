#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <omp.h>

// Macro for checking CUDA errors
#define CUDA_CHECK_ERROR() {                                           \
    cudaError_t err = cudaGetLastError();                              \
    if (err != cudaSuccess) {                                          \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)         \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(err);                                                     \
    }                                                                  \
}

// CUDA Kernel to rescale pixels on the GPU
__global__ void rescalePixels(short* inputImage, unsigned char* outputImage, int innerBound, int upperBound, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    if (idx < size) {
        short pixel = inputImage[idx];
        // Normalize and rescale the pixel value
        float normalized = (float)(pixel - innerBound) / (upperBound - innerBound);
        outputImage[idx] = (unsigned char)(normalized * 255.0f);
    }
}

int main() {
    // Query GPU information
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    CUDA_CHECK_ERROR();

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        std::cout << "Using GPU device " << device << ": " << deviceProp.name << std::endl;
    }

    // Number of pixels to process
    int size = 4000000;  // 4 million pixels

    // Allocate host memory
    short* h_image = new short[size];
    unsigned char* h_outputImage = new unsigned char[size];

    int innerBound = 1000;
    int upperBound = 30000;

    // Seed the random number generator
    srand(time(0));

    // Parallelized filling of the array with random values (using OpenMP)
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        h_image[i] = innerBound + rand() % (upperBound - innerBound + 1);
    }

    // Allocate device memory
    short* d_image;
    unsigned char* d_outputImage;
    cudaMalloc(&d_image, size * sizeof(short));
    CUDA_CHECK_ERROR();
    cudaMalloc(&d_outputImage, size * sizeof(unsigned char));
    CUDA_CHECK_ERROR();

    // Copy input data from host to device
    cudaMemcpy(d_image, h_image, size * sizeof(short), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    // Define block and grid sizes
    int blockSize = 512;
    int gridSize = (size + blockSize - 1) / blockSize;

    std::cout << "Number of blocks: " << gridSize << std::endl;
    std::cout << "Number of threads per block: " << blockSize << std::endl;
    std::cout << "Total number of threads: " << (gridSize * blockSize) << std::endl;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording time
    cudaEventRecord(start);

    // Launch the kernel with optimized grid and block size
    rescalePixels<<<gridSize, blockSize>>>(d_image, d_outputImage, innerBound, upperBound, size);
    CUDA_CHECK_ERROR();

    // Stop recording time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy output data from device to host
    cudaMemcpy(h_outputImage, d_outputImage, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();

    // Print the time taken
    std::cout << "Time taken to rescale " << size << " pixels on GPU: " << milliseconds << " milliseconds" << std::endl;

    // Free device memory
    cudaFree(d_image);
    CUDA_CHECK_ERROR();
    cudaFree(d_outputImage);
    CUDA_CHECK_ERROR();

    // Free host memory
    delete[] h_image;
    delete[] h_outputImage;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
