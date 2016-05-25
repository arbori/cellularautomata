#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
//#include <stdio.h>

/*!
*
*/
__global__ void BinarySiteEntropyKernel(int* ca, int X, int Y, float* entropyCA) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	float states[] = { 0.0, 0.0 };
	float sum = 0.0;

	for (int y = 0; y < Y; y++) {
		sum += 1.0;

		states[ca[x + y*X]] += 1.0;
	}

	states[0] /= sum;
	states[1] /= sum;

	if (states[0] == 0.0f || states[1] == 0.0f) {
		entropyCA[x] = 0.0f;
	}
	else {
		entropyCA[x] = -(states[0] * (log(states[0]) / log(2.0)) + states[1] * (log(states[1]) / log(2.0)));
	}
}

/*!
 *
 */
cudaError_t BinarySiteEntropy(int* ca, int X, int Y, float* entropyCA)
{
	int *dev_ca = 0;
	float *dev_entropyCA = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_ca, X * Y * sizeof(int));
	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_entropyCA, X * sizeof(float));
	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_ca, ca, X * Y * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	BinarySiteEntropyKernel << <1, X >> >(dev_ca, X, Y, dev_entropyCA);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(entropyCA, dev_entropyCA, X * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceReset failed!");
		goto Error;
	}

Error:
	cudaFree(dev_ca);
	cudaFree(dev_entropyCA);

	return cudaStatus;
}
