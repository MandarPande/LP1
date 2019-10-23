#include<iostream>
#include<chrono>
#include<cuda.h>
#include<cmath>
#include<cuda_runtime.h>
#define N 1024


using namespace std;
using namespace std::chrono;

static const int wholeArraySize = 10000000;
static const int blockSize = 1024;
static const int gridSize = 24; //this number is hardware-dependent; usually #SM*2 is a good number.

__global__ void sumCommMultiBlock(const int *gArr, int arraySize, int *gOut) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*blockSize;
    const int gridSize = blockSize*gridDim.x;
    int sum = 0;
    for (int i = gthIdx; i < arraySize; i += gridSize)
        sum += gArr[i];
    __shared__ int shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (thIdx<size)
            shArr[thIdx] += shArr[thIdx+size];
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
}

__global__ void numerator(const int *gArr, int arraySize, int *gOut, float mean) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*blockSize;
    const int gridSize = blockSize*gridDim.x;
    float sum = 0;
    for (int i = gthIdx; i < arraySize; i += gridSize)
        sum += (float(gArr[i]) - mean)*(float(gArr[i]) - mean);
    __shared__ float shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (thIdx<size)
            shArr[thIdx] += shArr[thIdx+size];
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
}


float stddev(int *arr, int n)
{
    float sum = 0.0f;
    for(int i = 0 ; i < n ; i++)
    {
        sum+=arr[i];
    }
    float mean = sum / n;
    sum = 0;
    for(int i = 0 ; i < n ; i++)
    {
        sum+=(arr[i] - mean)*(arr[i] - mean);
    }
    float res = sum/n;
    return float(sqrt(res));
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
    
    auto start1 = high_resolution_clock::now();
    sumCommMultiBlock<<<gridSize, blockSize>>>(dev_arr, wholeArraySize, dev_out);
    //dev_out now holds the partial result
    sumCommMultiBlock<<<1, blockSize>>>(dev_out, gridSize, dev_out);
    auto stop1 = high_resolution_clock::now();
    auto dur1 = duration_cast<microseconds>(stop1 - start1).count();
    //dev_out[0] now holds the final result
    cudaDeviceSynchronize();
    
    cudaMemcpy(&out, dev_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
    cudaFree(dev_out);
    cout<<"Sum is : "<<out;
    
    cudaMalloc((void**)&dev_arr, wholeArraySize * sizeof(int));
    cudaMemcpy(dev_arr, arr, wholeArraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&dev_out, sizeof(int)*gridSize);

    int sum = out;
    float mean = float(sum)/wholeArraySize;

    auto start2 = high_resolution_clock::now();
    numerator<<<gridSize, blockSize>>>(dev_arr, wholeArraySize, dev_out, mean);
    sumCommMultiBlock<<<1, blockSize>>>(dev_out, gridSize, dev_out);
    auto stop2 = high_resolution_clock::now();
    auto dur2 = duration_cast<microseconds>(stop2 - start2).count();
    cout<<"\nPARALLEL TIME : "<<dur1 + dur2<<endl;
    cudaDeviceSynchronize();
    
    cudaMemcpy(&out, dev_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
    cudaFree(dev_out);

    cout<<"numerator is : "<<out;

    float num = out;

    float term = num / wholeArraySize;

    float sol = sqrt(term);

    cout<<"\nSTD DEV is : "<<sol<<endl;
    start1 = high_resolution_clock::now();
    float sol2 = stddev(arr, wholeArraySize);
    stop1 = high_resolution_clock::now();
    dur1 = duration_cast<microseconds>(stop1 - start1).count();
    cout<<"SERIAL TIME is : "<<dur1<<endl;
    cout<<"\nSTD DEV is (SERIAL)  : "<<sol2;


    
}