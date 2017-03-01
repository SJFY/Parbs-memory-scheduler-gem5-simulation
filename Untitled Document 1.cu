#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#define BLOCK_SIZE 1024
__global__ void pi(int sample, curandState * state, unsigned long seed, float * input, float * output)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index<sample)
	{
        curand_init(seed, index, 0, &state[index]);
        float random = curand_uniform(&state[index]);
        curand_init(seed, index*2, 0, &state[index]);
        float random2 = curand_uniform(&state[index]);
        if (random*random + random2 * random2  < 1)
	{
	input[index] = 1;
	}
	else input[index] = 0;
        } 
	//now we start to use list reduction to calcualte the total number. 
	int len = sample;
	__shared__ float partialSum[BLOCK_SIZE*2];
        unsigned int t=threadIdx.x;
        unsigned int start=blockIdx.x*blockDim.x*2;
        if(start+t<len)
        {
           partialSum[t]=input[start+t];
        }
        else
                partialSum[t]=0;
        if(start+t+blockDim.x<len)
           partialSum[t+blockDim.x]=input[start+t+blockDim.x];
        else
                partialSum[t+blockDim.x]=0;

        __syncthreads();

        for(unsigned int stride=blockDim.x;stride>0;stride/=2)
        {
                __syncthreads();
                if(t<stride)
                        partialSum[t]+=partialSum[t+stride];
        }
        __syncthreads();
        //partialSum[0] is the sum of one block
        if(t==0)
        output[blockIdx.x]=partialSum[0];

}



int main(void)
{
    int host_sample = 30000;
    cudaError_t err = cudaSuccess;
    //Allocate the device random sequence state
    curandState* devStates;
    err = cudaMalloc(&devStates, host_sample*sizeof(curandState));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device random sequence state (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //Allocate the device output  
    float * devOutput;
    err = cudaMalloc((void**)&devOutput, host_sample*sizeof(float)); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //Allocate the device output of sum 
    float * devOutput2;
    err = cudaMalloc((void**)&devOutput2, host_sample*sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device output of sum (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Launch the pi Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid =ceil(host_sample/1024.0);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    pi<<<blocksPerGrid, threadsPerBlock>>>(host_sample, devStates, unsigned(time(NULL)), devOutput, devOutput2);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch pi kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copy device output sum vector back into host output sun vector
    printf("Copy output data from the CUDA device to the host memory\n");
    float hostOutput2[host_sample];
    err = cudaMemcpy(hostOutput2, devOutput2, host_sample*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy back device sum  output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    //Add the sum of each block, to get the total sum
    for (int i = 1; i < ceil(host_sample/2048.0); i++) 
    {
        hostOutput2[0] += hostOutput2[i];
    } 
    //Calculate pi
    float pi = 4 * hostOutput2[0]/ host_sample;
    printf("pi is %f.\n", pi);
    //Free device global memory
    err = cudaFree(devOutput);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(devOutput2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device sum output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(devStates);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device random sequence state (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Done.\n");



return 0;
}

