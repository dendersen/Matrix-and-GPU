#pragma once
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "source.h"

#define HIP_CHECK(status)                                                                                     \
    if (status != hipSuccess) {                                                                               \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(status), status, __FILE__, __LINE__); \
        exit(0);                                                                                              \
    }

extern uint8_t HIP_AVAILABLE;

__global__ void __GPU_Matrix_I_Add(int* a, int* b, int* target, uint32_t width, uint32_t height) {
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	target[y * width + x] = a[y * width + x] + b[y * width + x];
}

static Matrix_I_t* MatrixAdd_I_gpu(Matrix_I_t* a, Matrix_I_t* b) {
    if(!HIP_AVAILABLE) return MatrixAdd_I(a, b);
	if(a->width != b->width || a->height != b->height) return NULL;
    const uint32_t THREADS_PER_BLOCK_X = 4;
    const uint32_t THREADS_PER_BLOCK_Y = 4;
    const uint32_t THREADS_PER_BLOCK_Z = 1;
    const uint32_t GRID_X = uint32_t(a->width / THREADS_PER_BLOCK_X);
    const uint32_t GRID_Y = uint32_t(a->height / THREADS_PER_BLOCK_Y);
	
    dim3 dimGrid(GRID_X, GRID_Y);
    dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);

    int* matrixA = 0;
    int* matrixB = 0;
    int* target = 0;

    HIP_CHECK(hipMalloc((void**)&matrixA, a->width * a-> height * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&matrixB, a->width * a-> height * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&target, a->width * a-> height * sizeof(int)));

    HIP_CHECK(hipMemcpy(matrixA, a->data, a->width * a-> height * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(matrixB, b->data, a->width * a-> height * sizeof(int), hipMemcpyHostToDevice));

    __GPU_Matrix_I_Add <<<dimGrid, dimBlock, 0, 0>>>(matrixA, matrixB, target, a->width, a->height);

    Matrix_I_t* result = generateMatrix_I(a->width, a->height,false);
    
    HIP_CHECK(hipMemcpy(result->data, target, a->width * a-> height * sizeof(int), hipMemcpyDeviceToHost));
    
    HIP_CHECK(hipFree(target));
    HIP_CHECK(hipFree(matrixA));
    HIP_CHECK(hipFree(matrixB));

    return result;
}
