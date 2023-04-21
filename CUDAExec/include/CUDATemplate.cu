#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include "helper.h"

#include <cuda_profiler_api.h>

__global__ void test6(float* ptr, float* out)
{
    
    __shared__ float4 shareM[32];
    float4*           ptr_temp = (float4*)ptr;
    for (int i = 0; i < 10; i++) {
        if (threadIdx.x < 32) {
            shareM[threadIdx.x] = ptr_temp[threadIdx.x];
        }
        __syncwarp();
    }

    out[threadIdx.x] = ((float*)(shareM))[threadIdx.x];
}

int main(int argc, char** argv)
{
    size_t bytes = sizeof(float4) * 32;
    float *ptr, *out;
    CUDA_ERROR(cudaMalloc((void**)&ptr, bytes));
    CUDA_ERROR(cudaMalloc((void**)&out, bytes));

    CUDA_ERROR(cudaMemset(ptr, 0, bytes));
    CUDA_ERROR(cudaMemset(ptr, 1, bytes));

    CUDA_ERROR(cudaProfilerStart());

    test6<<<1, 32>>>(ptr, out);

    CUDA_ERROR(cudaDeviceSynchronize());

    CUDA_ERROR(cudaProfilerStop());

    return 0;
}
