#include <stdio.h>
#include <cuda.h>

#define N 1024
#define THREADS_PER_BLOCK 512

__global__ void reduce_sum(int *data) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    __shared__ int sdata[THREADS_PER_BLOCK];

    sdata[tid] = data[i];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        data[blockIdx.x] = sdata[0];
    }
}

int main() {
    int size = N * sizeof(int);
    int *h_data = (int *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_data[i] = 1;
    }

    int *d_data;
    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    int blocks = N / THREADS_PER_BLOCK;
    reduce_sum<<<blocks, THREADS_PER_BLOCK>>>(d_data);

    if (blocks > 1) {
        reduce_sum<<<1, blocks>>>(d_data);
    }

    int result;
    cudaMemcpy(&result, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum = %d\n", result);

    cudaFree(d_data);
    free(h_data);
    return 0;
}
