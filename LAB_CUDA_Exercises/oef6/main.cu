#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define WIDTH 128
#define HEIGHT 128
#define NUM_STREAMS 4

__global__ 
void imageProcessingKernel(int *image, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int index = y * width + x;
        // Simulated image processing task: setting pixel value to 1
        image[index] = 1;
    }
}

int main()
{
    int *d_image[NUM_STREAMS];
    int *h_image[NUM_STREAMS];
    cudaStream_t stream[NUM_STREAMS];
    
    size_t size = WIDTH * HEIGHT * sizeof(int);
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaMalloc((void **)&d_image[i], size);
        cudaMallocHost(&h_image[i], size);
        
        for (int j = 0; j < WIDTH * HEIGHT; ++j) {
            h_image[i][j] = 0; // Initialize image data
        }
        
        cudaStreamCreate(&stream[i]);
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaMemcpyAsync(d_image[i], h_image[i], size, cudaMemcpyHostToDevice, stream[i]);
    }
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        imageProcessingKernel<<<gridSize, blockSize, 0, stream[i]>>>(d_image[i], WIDTH, HEIGHT);
    }
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaMemcpyAsync(h_image[i], d_image[i], size, cudaMemcpyDeviceToHost, stream[i]);
    }
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(stream[i]);
    }
    
    // Print result (just printing non-zero elements for simplicity)
    for (int i = 0; i < NUM_STREAMS; ++i) {
        printf("Stream %d:\n", i);
        for (int j = 0; j < WIDTH * HEIGHT; ++j) {
            if (h_image[i][j] != 0) {
                printf("(%d, %d): %d\n", j % WIDTH, j / WIDTH, h_image[i][j]);
            }
        }
    }
    
    // Free memory and destroy streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaFree(d_image[i]);
        cudaFreeHost(h_image[i]);
        cudaStreamDestroy(stream[i]);
    }
    
    printf("\nDone\n");
    
    return 0;
}
