#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include "helper.h"

#include <cuda_profiler_api.h>

#define nthreads 32

struct myFloat4
{
    float x, y, z, w;
};

__global__ void test6(float* ptr, float* out)
{

    __shared__ myFloat4 shareM[nthreads];
    myFloat4*           ptr_temp = (myFloat4*)ptr;
    // for (int i = 0; i < 10; i++) {
    if (threadIdx.x < nthreads) {
        ((float*)(shareM))[threadIdx.x] = ((float*)(ptr_temp))[threadIdx.x];
        //printf("\n Read T=%d => %p", threadIdx.x, &shareM[threadIdx.x]);
    }
    __syncwarp();
    //}

    out[threadIdx.x] = ((float*)(shareM))[threadIdx.x];
    //printf(
    //    "\n Write T=%d => %p", threadIdx.x, &((float*)(shareM))[threadIdx.x]);
}

int main(int argc, char** argv)
{

    size_t bytes = sizeof(myFloat4) * nthreads;
    float *ptr, *out;
    CUDA_ERROR(cudaMalloc((void**)&ptr, bytes));
    CUDA_ERROR(cudaMalloc((void**)&out, bytes));

    CUDA_ERROR(cudaMemset(ptr, 0, bytes));
    CUDA_ERROR(cudaMemset(ptr, 1, bytes));

    CUDA_ERROR(cudaProfilerStart());

    test6<<<1, nthreads>>>(ptr, out);

    CUDA_ERROR(cudaDeviceSynchronize());

    CUDA_ERROR(cudaProfilerStop());

    return 0;
}
