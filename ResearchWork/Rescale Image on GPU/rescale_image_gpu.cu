#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <omp.h>

// Check if CUDA messed up (so we don't panic later)
#define CUDA_CHECK_ERROR() {                                           \
    cudaError_t err = cudaGetLastError();                              \
    if (err != cudaSuccess) {                                          \
        std::cerr << "Oops, CUDA goofed: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(err);                                                     \
    }                                                                  \
}

// This does the magic to rescale pixels using the GPU
__global__ void rescalePixels(short* inputImage, unsigned char* outputImage, int innerBound, int upperBound, int size) {
    // Get the thread index (the GPU runs many of these at once)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If we're within range, rescale the pixel
    if (idx < size) {
        short pixel = inputImage[idx];
        float normalized = (float)(pixel - innerBound) / (upperBound - innerBound);
        outputImage[idx] = (unsigned char)(normalized * 255.0f); // Make it between 0 and 255
    }
}

int main() {
    // How many GPUs are on this machine?
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    CUDA_CHECK_ERROR(); // Did it mess up

    // Loop through each GPU and print some info
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        std::cout << "Hey, we're using GPU " << device << ": " << deviceProp.name << std::endl;
    }

    // Let's process 4 million pixels 
    int size = 4000000;

    // Make some space in memory 
    short* h_image = new short[size];
    unsigned char* h_outputImage = new unsigned char[size];

    // These numbers are just for rescaling the pixels later
    int innerBound = 1000;
    int upperBound = 30000;

    // Fill the image with random pixel values
    srand(time(0));  // Randomness needs a seed

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        h_image[i] = innerBound + rand() % (upperBound - innerBound + 1);
    }

    // Now let's move to the GPU 
    short* d_image; // GPU memory for input
    unsigned char* d_outputImage; // GPU memory for output

    // Allocate space on the GPU 
    cudaMalloc(&d_image, size * sizeof(short));
    CUDA_CHECK_ERROR(); // Any errors so far
    cudaMalloc(&d_outputImage, size * sizeof(unsigned char));
    CUDA_CHECK_ERROR(); // Another error check 

    // Copy the data from CPU to GPU. now the GPU has the image
    cudaMemcpy(d_image, h_image, size * sizeof(short), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    // Time to define how many threads we'll run on the GPU 
    int blockSize = 1024;
    int gridSize = (size + blockSize - 1) / blockSize;  // Calculate how many blocks we need

    // Create some events to time the operation 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing the GPU magic
    cudaEventRecord(start);

    // Run the pixel rescaling function on the GPU 
    rescalePixels<<<gridSize, blockSize>>>(d_image, d_outputImage, innerBound, upperBound, size);
    CUDA_CHECK_ERROR();

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate how much time it took
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result back to CPU memory
    cudaMemcpy(h_outputImage, d_outputImage, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();

    // Print how long the GPU took 
    std::cout << "It took " << milliseconds << " milliseconds to rescale 4 million pixels!" << std::endl;

    // Clean up the GPU memory 
    cudaFree(d_image);
    CUDA_CHECK_ERROR();
    cudaFree(d_outputImage);
    CUDA_CHECK_ERROR();

    // Clean up the CPU memory
    delete[] h_image;
    delete[] h_outputImage;

    // Destroy the events 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
