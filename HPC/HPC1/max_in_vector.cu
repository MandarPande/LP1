#include<iostream>
#include<chrono>
#include<cuda.h>
#include<cuda_runtime.h>
#define N 1024


using namespace std;
using namespace std::chrono;

static const int wholeArraySize = 100000000;
static const int blockSize = 1024;
static const int gridSize = 24; //this number is hardware-dependent; usually #SM*2 is a good number.

__global__ void maxPerBlock(const int *gArr, int arraySize, int *gOut) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*blockSize;
    const int gridSize = blockSize*gridDim.x;
    int max = gArr[0];
    for (int i = gthIdx; i < arraySize; i += gridSize)
        if(max < gArr[i])
            max = gArr[i];
    __shared__ int shArr[blockSize];
    shArr[thIdx] = max;
    __syncthreads();
    /*for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (thIdx<size)
            shArr[thIdx] += shArr[thIdx+size];
        __syncthreads();
    }*/
    if (thIdx == 0)
    {
        max = shArr[0];
        for(int i = 0 ; i < blockSize ; i++)
        {
            if(max < shArr[i])
            {
                max = shArr[i];
            }
        }
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = max;
}


int main() {
    int *arr = new int[wholeArraySize];
    for(int  i =  0; i < wholeArraySize ; i++)
    {
        arr[i] = (i+1)%10;
    }
    int* dev_arr;
    cudaMalloc((void**)&dev_arr, wholeArraySize * sizeof(int));
    cudaMemcpy(dev_arr, arr, wholeArraySize * sizeof(int), cudaMemcpyHostToDevice);

    int out;
    int* dev_out;
    cudaMalloc((void**)&dev_out, sizeof(int)*gridSize);
    
    maxPerBlock<<<gridSize, blockSize>>>(dev_arr, wholeArraySize, dev_out);
    //dev_out now holds the partial result
    maxPerBlock<<<1, blockSize>>>(dev_out, gridSize, dev_out);
    //dev_out[0] now holds the final result
    cudaDeviceSynchronize();
    
    cudaMemcpy(&out, dev_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
    cudaFree(dev_out);
    cout<<"Max is : "<<out;

    
}
