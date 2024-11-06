#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <omp.h>
#include "ImageRescalerJNI.h"  // Include the JNI header file

#define CUDA_CHECK_ERROR() {                                           \
    cudaError_t err = cudaGetLastError();                              \
    if (err != cudaSuccess) {                                          \
        std::cerr << "Oops, CUDA goofed: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(err);                                                     \
    }                                                                  \
}

__global__ void rescalePixels(short* inputImage, unsigned char* outputImage, int innerBound, int upperBound, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        short pixel = inputImage[idx];
        float normalized = (float)(pixel - innerBound) / (upperBound - innerBound);
        outputImage[idx] = (unsigned char)(normalized * 255.0f);
    }
}

extern "C"
JNIEXPORT void JNICALL Java_ImageRescalerJNI_rescaleImage
  (JNIEnv *env, jobject thisObj, jshortArray jInputImage, jbyteArray jOutputImage, jint innerBound, jint upperBound, jint size) {

    // Get pointer to the input and output arrays from Java
    jshort* inputImage = env->GetShortArrayElements(jInputImage, NULL);
    jbyte* outputImage = env->GetByteArrayElements(jOutputImage, NULL);

    short* d_image;
    unsigned char* d_outputImage;

    cudaMalloc(&d_image, size * sizeof(short));
    CUDA_CHECK_ERROR();
    cudaMalloc(&d_outputImage, size * sizeof(unsigned char));
    CUDA_CHECK_ERROR();

    cudaMemcpy(d_image, inputImage, size * sizeof(short), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    rescalePixels<<<gridSize, blockSize>>>(d_image, d_outputImage, innerBound, upperBound, size);
    CUDA_CHECK_ERROR();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(outputImage, d_outputImage, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();

    std::cout << "GPU took " << milliseconds << " milliseconds to rescale the image!" << std::endl;

    cudaFree(d_image);
    cudaFree(d_outputImage);
    env->ReleaseShortArrayElements(jInputImage, inputImage, 0);
    env->ReleaseByteArrayElements(jOutputImage, outputImage, 0);
}
