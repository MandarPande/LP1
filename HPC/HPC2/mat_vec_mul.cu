#include "assignmentHPC2.cuh"

#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace std::chrono;

/*

Matrix Size : M x N
Vector Size : N x 1

*/

#define M 1024*1024
#define N 1024

#define BLOCKSIZE 32


__global__ void mat_vec_mul_kernel(float *a, float *b, float *c) {

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int tindex = tidx + gridDim.x * BLOCKSIZE * tidy;

    if(tindex < M) {
        int temp = tindex * N;
        for(int i = 0; i < N; i++) {
            c[tindex] += a[temp + i] * b[i];
        }
    }
    __syncthreads();

}


void mat_vec_mul_cpu(float *a, float *b, float *c) {

    for(unsigned int i = 0; i < M; i++) {
        for(unsigned int j = 0; j < N; j++) {
            c[i] += a[i*N + j] * b[j];
        }
    }

}

int main() {

    // declare variables
    float *a_host, *b_host, *c_host;
    float *a_device, *b_device, *c_device;

    // allocate memory to host variables
    a_host = (float *)malloc(M * N * sizeof(float));
    b_host = (float *)malloc(N * sizeof(float));
    c_host = (float *)malloc(M * sizeof(float));

    // initialize host variables
    for(int i = 0; i < M*N; i++) {
        a_host[i] = 1.0f ;//1024*1024 * float(rand())/RAND_MAX;
    }

    for(int i = 0; i < N; i++) {
        b_host[i] = 1.0f ;//1024*1024 * float(rand())/RAND_MAX;
    }


    cout<<"INPUT SIZE "<<endl;
    cout<<"Matrix A : "<<M<<" * "<<N<<endl;
    cout<<"Vector B : "<<N<<" * "<<1<<endl;
   


    // ----------------------------------------- CPU Code -------------------------------------------------
    memset(c_host, 0, M * sizeof(float));

    // call vec_add_cpu function
    auto startCPU = high_resolution_clock::now();
    mat_vec_mul_cpu(a_host, b_host, c_host);
    auto stopCPU = high_resolution_clock::now();

    // Display Results
    cout<<"\n\n--------------- CPU ---------------\n"<<endl;
    cout<<"Answer CPU : \n"<<endl;
    for(int i = 0; i < 5; i++) {
        cout<<c_host[i]<<endl;
    }
    cout<<"\nTime on CPU : "<<duration_cast<microseconds>(stopCPU - startCPU).count()/1000<<" milli seconds\n\n"<<endl;

    free(c_host);



    // ----------------------------------------- GPU Code -------------------------------------------------


    // allocate memory to device vairables
    cudaMalloc(&a_device, M * N * sizeof(float));
    cudaMalloc(&b_device, N * sizeof(float));
    cudaMalloc(&c_device, M * sizeof(float));

    // copy data from host to device
    cudaMemcpy(a_device, a_host, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(c_device, 2, M * sizeof(float));

    // set up timing variables
	float gpu_elapsed_time;
	cudaEvent_t gpu_start, gpu_stop;

	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);


	// call kernel
    cudaEventRecord(gpu_start, 0);    

    // call Kernel
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(1, M / (BLOCKSIZE*BLOCKSIZE) + 1);

    auto startGPU = high_resolution_clock::now();
    mat_vec_mul_kernel<<<gridSize, blockSize>>>(a_device, b_device, c_device);

    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    // copy results from device to host
    c_host = (float *)malloc(M * sizeof(float));
    cudaMemcpy(c_host, c_device, M * sizeof(float), cudaMemcpyDeviceToHost);

    // Display Results
    cout<<"--------------- GPU ---------------\n"<<endl;
    cout<<"Answer GPU : \n"<<endl;
    for(int i = 0; i < 5; i++) {
        cout<<c_host[i]<<endl;
    }
    cout<<"\nTime on GPU : "<<gpu_elapsed_time<<" milli seconds\n\n"<<endl;

    // Free allocated Memory
    free(a_host);
    free(b_host);
    free(c_host);
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
    return 0;
}
