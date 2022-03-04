#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <coreutils/functions/debug/print.cpp>
#include <coreutils/util/time.hpp>
#include <coreutils/util/cudaErrors.cuh>
#include <iostream>
#include <sys/resource.h>

using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::debug;

// this program starts by consolidating each of the blocks through each of the threads
// and then putting those into corresponding values in g_odata
// 
// g_idata corresponds to the input data
// g_odata corresponds to the output data
//	n corresponds to the maximum value that i should reach. same as the number of elements
// gridSize is how much should be added to i to get to the next value
// assumes that shared size is 32k or less
template <unsigned int blockSize>
__global__ void summationOfArrays(float *g_idata, float* g_odata, int n){
	// tells the gpu that there is some shared data out there
	extern __shared__ float sdata[];

	// i symbolizes how many times the addition should be done.
	// starts at blockIdx.x * blockSize * 2 (getting to the actual block) + tid (getting to the actual thread in the actual block)
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	
	while (i < n) {
		sdata[tid] += g_idata[i] + g_idata[i+blockSize];
		i += gridSize;
	}
	if (i - gridSize + blockSize >= n) {
		sdata[tid] -= g_idata[i+blockSize];
	}
	__syncthreads();


	// add each of the threads together
	if (tid < 512) { if (blockSize >= 1024) { sdata[tid] += sdata[tid + 512]; __syncthreads(); }}
	if (tid < 256) { if (blockSize >= 512) { sdata[tid] += sdata[tid + 256]; __syncthreads(); }}
	if (tid < 128) { if (blockSize >= 256) { sdata[tid] += sdata[tid + 128];  __syncthreads(); }}
	if (tid < 64) { if (blockSize >= 128) { sdata[tid] += sdata[tid + 64];  __syncthreads(); }}
	if (tid < 32) { if (blockSize >= 64) {sdata[tid] += sdata[tid + 32]; __syncthreads(); }}
	if (tid < 16) { if (blockSize >= 32) {sdata[tid] += sdata[tid + 16]; __syncthreads(); }}
	if (tid < 8) { if (blockSize >= 16) {sdata[tid] += sdata[tid + 8]; __syncthreads(); }}
	if (tid < 4) {	if (blockSize >= 8) {sdata[tid] += sdata[tid + 4]; __syncthreads(); }
		if (blockSize >= 4) {sdata[tid] += sdata[tid + 2]; __syncthreads(); }
		if (blockSize >= 2) {sdata[tid] += sdata[tid + 1]; }
	}
	
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
	}
}

template <unsigned int blockSize>
__global__ void multiplyAndAddArrays (float* g_idata1, float* g_idata2, float* g_odata, int n) {
	// tells the gpu that there is some shared data out there
	extern __shared__ float sdata[];

	// i symbolizes how many times the addition should be done.
	// starts at blockIdx.x * blockSize * 2 (getting to the actual block) + tid (getting to the actual thread in the actual block)
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	
	while (i < n) {
		sdata[tid] += g_idata1[i] * g_idata2[i] + g_idata1[i+blockSize] * g_idata2[i+blockSize];
		// sdata[tid] += g_idata1[i] + g_idata1[i+blockSize];
		i += gridSize;
	}
	if (i - gridSize + blockSize >= n) {
		sdata[tid] -= g_idata1[i+blockSize] * g_idata2[i+blockSize];
		// sdata[tid] -= g_idata1[i+blockSize];
	}
	__syncthreads();

	// add each of the threads together
	// if (tid < 512) {
	// 	if (blockSize >= 1024) { sdata[tid] += sdata[tid + 512]; __syncthreads(); }
	// 	if (blockSize >= 512) { sdata[tid] += sdata[tid + 256]; __syncthreads(); }
	// 	if (blockSize >= 256) { sdata[tid] += sdata[tid + 128];  __syncthreads(); }
	// 	if (blockSize >= 128) { sdata[tid] += sdata[tid + 64];  __syncthreads(); }
	// 	if (blockSize >= 64) { sdata[tid] += sdata[tid + 32]; __syncthreads(); }
	// 	if (blockSize >= 32) { sdata[tid] += sdata[tid + 16]; __syncthreads(); }
	// 	if (blockSize >= 16) { sdata[tid] += sdata[tid + 8]; __syncthreads(); }
	// 	if (blockSize >= 8) { sdata[tid] += sdata[tid + 4]; __syncthreads(); }
	// 	if (blockSize >= 4) { sdata[tid] += sdata[tid + 2]; __syncthreads(); }
	// 	if (blockSize >= 2) { sdata[tid] += sdata[tid + 1]; }
	// }
	if (tid < 512) { if (blockSize >= 1024) { sdata[tid] += sdata[tid + 512]; __syncthreads(); }}
	if (tid < 256) { if (blockSize >= 512) { sdata[tid] += sdata[tid + 256]; __syncthreads(); }}
	if (tid < 128) { if (blockSize >= 256) { sdata[tid] += sdata[tid + 128];  __syncthreads(); }}
	if (tid < 64) { if (blockSize >= 128) { sdata[tid] += sdata[tid + 64];  __syncthreads(); }}
	if (tid < 32) { if (blockSize >= 64) {sdata[tid] += sdata[tid + 32]; __syncthreads(); }}
	if (tid < 16) { if (blockSize >= 32) {sdata[tid] += sdata[tid + 16]; __syncthreads(); }}
	if (tid < 8) { if (blockSize >= 16) {sdata[tid] += sdata[tid + 8]; __syncthreads(); }}
	if (tid < 4) {	if (blockSize >= 8) {sdata[tid] += sdata[tid + 4]; __syncthreads(); }
		if (blockSize >= 4) {sdata[tid] += sdata[tid + 2]; __syncthreads(); }
		if (blockSize >= 2) {sdata[tid] += sdata[tid + 1]; }
	}
	
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
	}
}

// host code for cuda
float sumMultiplyOfMatrixesHost(Matrix3D* first, Matrix3D* second, int numBlocks, int threads, int numPerThread, double& returnTime, double& totalGpuTime) {
	if (first->getSize() != second->getSize()) {
		std::cout << "Invalid sizes in sum multiply matrixes.\n";
		exit(1);
	}
	totalGpuTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;

	// defining pointers and main variables
	long long size = first->getSize() / sizeof(float);
	int maxArrayIndex = numBlocks * threads * numPerThread;
	int sharedSize = threads * sizeof(float);
	int sizeLeft = size;
	float sum = 0;

	float* firstArr = first->getArr();
	float* secondArr = second->getArr();
	float* output = new float[numBlocks];
	float* device_firstArr;
	float* device_secondArr;
	float* device_output;

	// std::cout << first->getSize() << '\n';
	// exit(0);


	double startTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	
	gpuErrchk(cudaMalloc((void **) &device_firstArr, maxArrayIndex * sizeof(float)));
	gpuErrchk(cudaMalloc((void **) &device_secondArr, maxArrayIndex * sizeof(float)));
	gpuErrchk(cudaMalloc((void **) &device_output, numBlocks * sizeof(float)));

	double finalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - startTime;
	// std::cout << "Time to copy memory: " << std::fixed << finalTime << "s\n";

	// std::cout << "Number of threaTime to complete:ds: " << threads << '\n';
	// std::cout << "Number of blocks: " << numBlocks << '\n';
	// std::cout << "Number of bytes for shared storage: " << sharedSize << "\n";
	// std::cout << "Max array index: " << maxArrayIndex << "\n";
	// std::cout << "Max byte index: " << maxArrayIndex * sizeof(float) << "\n";
 
	double totalTime = 0;
sumMultiplyOfMatrixesHost
		// std::cout << "Sum should be " << (size - sizeLeft) * 0.05 << "\n";
		// std::cout << "Sum actually is " << sum << '\n';
		gpuErrchk(cudaMemcpy(device_firstArr, &firstArr[size - sizeLeft], maxArrayIndex * sizeof(float), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(device_secondArr, &secondArr[size - sizeLeft], maxArrayIndex * sizeof(float), cudaMemcpyHostToDevice));
		startTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
		
		switch (threads){
			case 1024:
				multiplyAndAddArrays<1024><<< numBlocks, threads, sharedSize >>>(device_firstArr, device_secondArr, device_output, maxArrayIndex); break;
			case 512:
				multiplyAndAddArrays<512><<< numBlocks, threads, sharedSize >>>(device_firstArr, device_secondArr, device_output, maxArrayIndex); break;
			case 256:
				multiplyAndAddArrays<256><<< numBlocks, threads, sharedSize >>>(device_firstArr, device_secondArr, device_output, maxArrayIndex); break;
			case 128:
				multiplyAndAddArrays<128><<< numBlocks, threads, sharedSize >>>(device_firstArr, device_secondArr, device_output, maxArrayIndex); break;
			case 64:
				multiplyAndAddArrays<64><<< numBlocks, threads, sharedSize >>>(device_firstArr, device_secondArr, device_output, maxArrayIndex); break;
			case 32:
				multiplyAndAddArrays<32><<< numBlocks, threads, sharedSize >>>(device_firstArr, device_secondArr, device_output, maxArrayIndex); break;
			case 16:
				multiplyAndAddArrays<16><<< numBlocks, threads, sharedSize >>>(device_firstArr, device_secondArr, device_output, maxArrayIndex); break;
			case 8:
				multiplyAndAddArrays<8><<< numBlocks, threads, sharedSize >>>(device_firstArr, device_secondArr, device_output, maxArrayIndex); break;
			case 4:
				multiplyAndAddArrays<4><<< numBlocks, threads, sharedSize >>>(device_firstArr, device_secondArr, device_output, maxArrayIndex); break;
			case 2:
				multiplyAndAddArrays<2><<< numBlocks, threads, sharedSize >>>(device_firstArr, device_secondArr, device_output, maxArrayIndex); break;
			case 1:
				multiplyAndAddArrays<1><<< numBlocks, threads, sharedSize >>>(device_firstArr, device_secondArr, device_output, maxArrayIndex); break;
		}
		
		finalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - startTime;
		totalTime += finalTime;
		// std::cout << "Time to run kernel: " << std::fixed << finalTime << "s\n";
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(output, device_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
		sizeLeft -= maxArrayIndex;
		for (int i = 0; i < numBlocks; i++){
			sum += output[i];
		}
	} while (sizeLeft > 0);

	returnTime = totalTime;
	cudaFree(device_firstArr);
	cudaFree(device_secondArr);
	cudaFree(device_output);
	delete[] output;

	totalGpuTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000 - totalGpuTime;

	return sum;
}

float summationTest(Matrix3D* m1, int numBlocks, int threads, int numPerThread, double& returnTime, double& totalGpuTime) {
	totalGpuTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;

	long long size = m1->getSize() / sizeof(float);
	int maxArrayIndex = numBlocks * threads * numPerThread;
	int sharedSize = threads * sizeof(float);
	int sizeLeft = size;
	float sum = 0;

	float* input = m1->getArr();
	float* output = new float[numBlocks];
	float* device_input;
	float* device_output;
	
	gpuErrchk(cudaMalloc((void **) &device_input, maxArrayIndex * sizeof(float)));
	gpuErrchk(cudaMalloc((void **) &device_output, numBlocks * sizeof(float)));

	// std::cout << "Number of threads: " << threads << '\n';
	// std::cout << "Number of blocks: " << numBlocks << '\n';
	// std::cout << "Number of bytes for shared storage: " << sharedSize << "\n";
	// std::cout << "Max array index: " << maxArrayIndex << "\n";
	// std::cout << "Max byte index: " << maxArrayIndex * sizeof(float) << "\n";
	double startTime = 0;
	double finalTime = 0;
	double totalTime = 0;
	do {
		if (sizeLeft - maxArrayIndex < 0) {
			maxArrayIndex = sizeLeft;
		}
		// std::cout << "Max array index " << maxArrayIndex << "\n";
		// std::cout << "Size left " << sizeLeft << "\n";
		// std::cout << "Sum should be " << size - sizeLeft << "\n";
		// std::cout << "Sum actually is " << sum << '\n';
		gpuErrchk(cudaMemcpy(device_input, &input[size - sizeLeft], maxArrayIndex * sizeof(float), cudaMemcpyHostToDevice));
		// std::cout << "Time to copy memory: " << std::fixed << finalTime << "s\n";

		startTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;

		switch (threads){
			case 1024:
				summationOfArrays<1024><<< numBlocks, threads, sharedSize >>>(device_input, device_output, maxArrayIndex); break;
			case 512:
				summationOfArrays<512><<< numBlocks, threads, sharedSize >>>(device_input, device_output, maxArrayIndex); break;
			case 256:
				summationOfArrays<256><<< numBlocks, threads, sharedSize >>>(device_input, device_output, maxArrayIndex); break;
			case 128:
				summationOfArrays<128><<< numBlocks, threads, sharedSize >>>(device_input, device_output, maxArrayIndex); break;
			case 64:
				summationOfArrays<64><<< numBlocks, threads, sharedSize >>>(device_input, device_output, maxArrayIndex); break;
			case 32:
				summationOfArrays<32><<< numBlocks, threads, sharedSize >>>(device_input, device_output, maxArrayIndex); break;
			case 16:
				summationOfArrays<16><<< numBlocks, threads, sharedSize >>>(device_input, device_output, maxArrayIndex); break;
			case 8:
				summationOfArrays<8><<< numBlocks, threads, sharedSize >>>(device_input, device_output, maxArrayIndex); break;
			case 4:
				summationOfArrays<4><<< numBlocks, threads, sharedSize >>>(device_input, device_output, maxArrayIndex); break;
			case 2:
				summationOfArrays<2><<< numBlocks, threads, sharedSize >>>(device_input, device_output, maxArrayIndex); break;
			case 1:
				summationOfArrays<1><<< numBlocks, threads, sharedSize >>>(device_input, device_output, maxArrayIndex); break;
		}

		// std::cout << "Time to run kernel: " << std::fixed << finalTime << "s\n";

		gpuErrchk(cudaDeviceSynchronize());
		finalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - startTime;
		totalTime += finalTime;
		gpuErrchk(cudaMemcpy(output, device_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
		sizeLeft -= maxArrayIndex;
		for (int i = 0; i < numBlocks; i++){
			sum += output[i];
		}
		// printArr(output, numBlocks);

	} while (sizeLeft > 0);
	// std::cout << "Time to run kernel: " << std::fixed << totalTime << "s\n";
	returnTime = totalTime;
	cudaFree(device_input);
	cudaFree(device_output);
	delete[] output;

	totalGpuTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000 - totalGpuTime;

	return sum;
}

int main () {
	int l = 150;
	int w = 1000;
	int h = 1000;
	Matrix3D* m1 = new Matrix3D(l, w, h);
	Matrix3D* m2 = new Matrix3D(l, w, h);

	for (int i = 0; i < l; i++) {
		for (int k = 0; k < w; k++) {
			for (int j = 0; j < h; j++) {
				m1->insert(1, i, k, j);
				m2->insert(0.1, i, k, j);
			}
		}
	}

	// m1->randomize();
	// m2->randomize();
	int count = 1;
	
	float val3, val4;

	double startTime = 0;
	double finalTime2 = 0;
	double finalTime3 = 0;

	std::cout << "\n\n\n:::STARTING CPU TESTS:::\n\n";
	startTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	for (int i = 0; i < count; i++) {
		Matrix3D* m3 = *m1 * m2;
		val3 = m3->sum();
		finalTime2 = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - startTime;

		startTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
		val4 = m1->sum();
		delete m3;
	}
	std::cout << std::fixed << "Final sum: " << val3 << '\n';
	finalTime3 = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - startTime;
	std::cout << "Time to Complete CPU: " << std::fixed << finalTime2 << "s\n\n\n\n";


	double trueMultiply = val3;
	double trueSum = val4;

	double finalTime = 100;
	double totalGpuTime = 100;
	double totalGpuTime2 = 100;

	float val = 0;
	float val2;

	int sk, sj, sl;
	double smallestTime = 100;
	double closestVal = 0;
	int sk2, sj2, sl2;
	double smallestTime2 = 100;
	double closestVal2 = 0;
	int sk3, sj3, sl3;
	double smallestTime3 = 100;
	double closestVal3 = 0;

	std::cout << std::fixed;
	std::cout.precision(7);

	std::cout << ":::STARTING GPU TESTS:::\n\n";
	std::cout << ":::STARTING ADDITION TESTS:::\n\n";
	for (int i = 0; i < count; i++) {
		for (int k = 16; k <= 16384; k *= 2) {
			for (int j = 32; j <= 1024; j *= 2) {
				for (int l = 8; l <= 4096; l *= 2) {
					if (((long long)k) * j * l <= 33554432 && k * j < 262144) {
						val2 = summationTest(m1, k, j, l, finalTime, totalGpuTime);
						printf("Blocks: %5d  Threads: %5d  NumPerThread: %5d  Kernel Time: %10.6f  GPU Time: %10.6f  Sum: %10.6f\n", k, j, l, finalTime, totalGpuTime, val2);
						if (finalTime < smallestTime3 && finalTime > 0) {
							closestVal3 = val2;
							smallestTime3 = finalTime;
							sk3 = k;
							sj3 = j;
							sl3 = l;
						}
						// std::cout.precision(9);
						// std::cout << "Time to Complete GPU: " << std::fixed << finalTime << "s\n";
					}
				}
			}
		}
	}

	std::cout << "\n\n\n:::STARTING MULTIPLICATION AND ADD TESTS:::\n\n\n\n";
	for (int i = 0; i < count; i++) {
		for (int k = 16; k <= 16384; k *= 2) {
			for (int j = 32; j <= 1024; j *= 2) {
				for (int l = 8; l <= 4096; l *= 2) {
					if (((long long)k) * j * l <= 33554432 && k * j < 262144) {
						val = sumMultiplyOfMatrixesHost (m1, m2, k, j, l, finalTime, totalGpuTime2);
						printf("Blocks: %5d  Threads: %5d  NumPerThread: %5d  Kernel Time: %10.6f  GPU Time: %10.6f  Sum: %10.6f\n", k, j, l, finalTime, totalGpuTime2, val);
						if (finalTime < smallestTime && finalTime > 0) {
							closestVal = val;
							smallestTime = finalTime;
							sk = k;
							sj = j;
							sl = l;
						}
						if ((trueMultiply - val) / 1000.0 * (trueMultiply - val) <= (trueMultiply - closestVal2) / 1000.0 * (trueMultiply - closestVal2)) {
							smallestTime2 = finalTime;
							closestVal2 = val;
							sk2 = k;
							sj2 = j;
							sl2 = l;
						}
						// std::cout.precision(9);
						// std::cout << "Time to Complete GPU: " << std::fixed << finalTime << "s\n";
					}
				}
			}
		}
	}

	std::cout << "\n\n\n:::COMPUTATIONS COMPLETED:::\n" << m1->getSize() / sizeof(float) << "\n\n";
	
	printf(":::CPU TIME TO ADD:::\nBlocks: %d\nThreads: %d\nNumPerThread: %llu\nSum: %f\nTime to complete: %fs\nAccuracy: %f percent\n\n\n", 1, 1,  m1->getSize() / sizeof(float), trueSum, finalTime3, 100.0);
	printf(":::CPU TIME TO MULTIPLY AND ADD:::\nBlocks: %d\nThreads: %d\nNumPerThread: %llu\nSum: %f\nTime to complete: %fs\nAccuracy: %f percent\n\n\n", 1, 1,  m1->getSize() / sizeof(float), trueMultiply, finalTime2, 100.0);

	printf(":::GPU BEST TIME TO ADD:::\nBlocks: %d\nThreads: %d\nNumPerThread: %d\nSum: %f\nTime to complete: %fs\nAccuracy: %f percent\n\n", sk3, sj3, sl3, closestVal3, smallestTime3, (closestVal3 / trueSum) * 100);
	printf(":::GPU BEST TIME TO MULTIPLY AND ADD:::\nBlocks: %d\nThreads: %d\nNumPerThread: %d\nSum: %f\nKernel Time: %fs\nGPU time: %fs\nAccuracy: %f percent\n\n\n", sk, sj, sl, closestVal, smallestTime, totalGpuTime, (closestVal / trueMultiply) * 100 > 100 ? (trueMultiply / closestVal * 100) : (closestVal / trueMultiply) * 100);
	printf(":::GPU BEST ACCURACY TO MULTIPLY AND ADD:::\nBlocks: %d\nThreads: %d\nNumPerThread: %d\nSum: %f\nKernel Time: %fs\nGPU time: %fs\nAccuracy: %f percent\n\n\n", sk2, sj2, sl2, closestVal2, smallestTime2, totalGpuTime2, (closestVal2 / trueMultiply) * 100 > 100 ? (trueMultiply / closestVal2 * 100) : (closestVal2 / trueMultiply) * 100);
	


	struct rusage usage;
   getrusage (RUSAGE_SELF, &usage);
   std::cout << "\nMemory used (MB) GPU: " << usage.ru_maxrss / 1000000 << "\n\n";
}