#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <sys/resource.h>

#include <coreutils/classes/matrixes/Matrix3D.cpp>
#include <artificialIntelligence/basicLearningTypes/generationalAIBasic.hpp>
#include <artificialIntelligence/basicLearningTypes/generationalAIBasic.cuh>
#include <artificialIntelligence/functions/activationFunctions.hpp>

#include <coreutils/functions/math/simpleMath.hpp>
#include <coreutils/functions/sort/sortHelpers.cpp>
#include <coreutils/functions/debug/print.cpp>

#include <artificialIntelligence/classes/BasicLayer.hpp>
#include <artificialIntelligence/classes/BasicLayerList.hpp>

#include <device_launch_parameters.h>

using namespace coreutils::classes::matrixes;
using namespace coreutils::functions;
using namespace artificialIntelligence::classes;
using namespace artificialIntelligence::functions::activation;
using namespace std;

namespace artificialIntelligence {
   namespace basicLearningTypes {
      namespace generationalAIBasic {
			
			// gpu error checking
			#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
			inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
			{
				if (code != cudaSuccess) 
				{
					fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
					if (abort) exit(code);
				}
			}

			// template <unsigned int blockSize>
			// __global__ void reduce6(int *g_idata, int *g_odata, unsigned int n)
			// {
			// extern __shared__ int sdata[];
			// unsigned int tid = threadIdx.x;
			// unsigned int i = blockIdx.x*(blockSize*2) + tid;
			// unsigned int gridSize = blockSize*2*gridDim.x;
			// sdata[tid] = 0;
			// while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
			// __syncthreads();
			// if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
			// if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
			// if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
			// if (tid < 32) {
			// 	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
			// 	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
			// 	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
			// 	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
			// 	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
			// 	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
			// }
			// if (tid == 0) g_odata[blockIdx.x] = sdata[0];
			// }

			// addition of matrixes
			// https://cuvilib.com/Reduction.pdf
			template <unsigned int blockSize>
			__global__ void summationOfArrays(float *g_idata, float* g_odata, unsigned int n){
				extern __shared__ float sdata[];
				unsigned int tid = threadIdx.x;
				unsigned int i = blockIdx.x*(blockSize*2) + tid;
				unsigned int gridSize = blockSize*2*gridDim.x;
				sdata[tid] = 0;

				while (i < n) {
					sdata[tid] += g_idata[i] + g_idata[i + blockSize];
					i += gridSize;
				}
				__syncthreads();

				if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
				if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
				if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

				if (tid < 32) {
					if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
					if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
					if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
					if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
					if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
					if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
				}

				if (tid == 0) g_odata[blockIdx.x] = sdata[0];
			}

			// multiplication of matrixes
			__global__ void multiplyArrays (float* in1, float* in2, float* out1) {
				out1[threadIdx.x] = in1[threadIdx.x] * in2[threadIdx.x];
			}
			
			// host code for cuda
			float sumMultiplyOfMatrixesHost(Matrix3D* first, Matrix3D* second) {
				float* weights = first->getArr();
				float* delta = second->getArr();
				int size = first->getSize() / sizeof(float);

				float* device_weights;
				float* device_delta;
				float* output = new float[size];
				float* device_output;

				// multiply
				gpuErrchk(cudaMalloc((void **) &device_weights, first->getSize()));
   			gpuErrchk(cudaMemcpy(device_weights, first->getArr(), first->getSize(), cudaMemcpyHostToDevice));

				gpuErrchk(cudaMalloc((void **) &device_delta, second->getSize()));
   			gpuErrchk(cudaMemcpy(device_delta, second->getArr(), second->getSize(), cudaMemcpyHostToDevice));

				gpuErrchk(cudaMalloc((void **) &device_output, size));
				
				multiplyArrays<<<1, size>>>(device_weights, device_delta, device_output);

				gpuErrchk(cudaDeviceSynchronize());
				gpuErrchk(cudaMemcpy(output, device_output, size, cudaMemcpyDeviceToHost));

				cudaFree(device_weights);
				cudaFree(device_delta);

				// use device_output as the same output so memory doesnt need to be reallocated
				// cudaFree(device_output);
				float* input = output;
				float* device_input;

				gpuErrchk(cudaMalloc((void **) &device_input, first->getSize()));
   			gpuErrchk(cudaMemcpy(device_input, input, first->getSize(), cudaMemcpyHostToDevice));

				
				// this chunk adds a each of the threads in each block together, however each block needs to be added together as well
				// < -- > //
				
				// 64-256 blocks, 128 threads, 1024-4096 elements per thread
				int threads = 128;

				int n = 1024;

				int numBlocks = size / threads / n;

				if (numBlocks == 0) {
					numBlocks = 1;
				}

				switch (threads){
					case 512:
						summationOfArrays<512><<< numBlocks, threads, first->getSize() >>>(device_input, device_output, n); break;
					case 256:
						summationOfArrays<256><<< numBlocks, threads, first->getSize() >>>(device_input, device_output, n); break;
					case 128:
						summationOfArrays<128><<< numBlocks, threads, first->getSize() >>>(device_input, device_output, n); break;
					case 64:
						summationOfArrays<64><<< numBlocks, threads, first->getSize() >>>(device_input, device_output, n); break;
					case 32:
						summationOfArrays<32><<< numBlocks, threads, first->getSize() >>>(device_input, device_output, n); break;
					case 16:
						summationOfArrays<16><<< numBlocks, threads, first->getSize() >>>(device_input, device_output, n); break;
					case 8:
						summationOfArrays<8><<< numBlocks, threads, first->getSize() >>>(device_input, device_output, n); break;
					case 4:
						summationOfArrays<4><<< numBlocks, threads, first->getSize() >>>(device_input, device_output, n); break;
					case 2:
						summationOfArrays<2><<< numBlocks, threads, first->getSize() >>>(device_input, device_output, n); break;
					case 1:
						summationOfArrays<1><<< numBlocks, threads, first->getSize() >>>(device_input, device_output, n); break;
				}

				exit(0);
				return *output;
			}

         void multiplyMatrixes(Matrix3D* first, Matrix3D* second, Matrix3D* output) {
         
         }

         void run (BasicLayerList* list, int epochs, double learningRate, Matrix3D** inputDataMatrixes, Matrix3D** outputDataMatrixes, int inputCount, bool calculateError, bool print) {

            // initial error

            std::cout.precision(4);

            double sumInitial = 0;
            if (calculateError) {
               for (int i = 0; i < inputCount; i++) {
                  list->editRootMatrix(inputDataMatrixes[i]);
                  list->calculateAndUpdateAll();
                  Matrix3D* error = *outputDataMatrixes[i] - list->getLast()->getLayerMatrix();
                  Matrix3D* squared = *error * error;
                  sumInitial += squared->sum() * 100;
                  delete error;
                  delete squared;
               }
               std::cout << "Total initial error :: " << sumInitial << "%\n\n";
            }
				
            int* order = new int[inputCount];
            for (int i = 0; i < inputCount; i++) {
               order[i] = i;
            }
   
            // main loop

            std::cout << std::fixed;
				std::cout.precision(2);
            // exit (0);
            for (int e = 0; e < epochs; e++) {
               // because stochastic gradient descent, the order needs randomization

               sort::shuffle(order, inputCount);
               
               std::cout << e / (double) epochs * 100 << "%\n";
                
               if (calculateError) {
                  float currentError = 0;
                  for (int i = 0; i < inputCount; i++) {
                     list->editRootMatrix(inputDataMatrixes[i]);
                     list->calculateAndUpdateAll();
						}
                  std::cout << "Total error :: " << currentError << "%\n\n";
               }
               // std::cout << std::setprecision(4);
               // double sum = 0;
               // for (int i = 0; i < inputCount; i++) {
               //    list->editRootMatrix(inputDataMatrixes[i]);
               //    list->calculateAndUpdateAll();
                  
               //    // (*outputDataMatrixes[i] -list->getLast()->getLayerMatrix())->printMatrix();
               //    // exit (0);
               //    std::cout << *outputDataMatrixes[i]->getData(0, 0, 0) << " :: " << *list->getLast()->getLayerMatrix()->getData(0, 0, 0) << " :: " << (*outputDataMatrixes[i] - list->getLast()->getLayerMatrix())->sum() * 100;
               //    sum += (*outputDataMatrixes[i] - list->getLast()->getLayerMatrix())->sum() * 100 > 0 ? (*outputDataMatrixes[i] - list->getLast()->getLayerMatrix())->sum() * 100 : (*outputDataMatrixes[i] - list->getLast()->getLayerMatrix())->sum() * 100 * -1;
               //    std::cout << "%" << " error\n";
               //    // list->getLast()->getLayerMatrix()->printMatrix();
               // }
               // std::cout << "Total error :: " << sum << "%";
               // std::cout << std::setprecision(2);
               // list->print(true, true);
               // std::cout << "\n\n";

               // debug::printArr(order, inputCount);
               // for (int i = 0; i < inputCount; i++) {
               //    cout << "Input Matrixes " << i << ":";
               //    inputDataMatrixes[order[i]]->printMatrix();
               //    cout << "Output Matrix " << i << ":";
               //    outputDataMatrixes[order[i]]->printMatrix();
               // }
               // exit (0);
               for (int i = 0; i < inputCount; i++) {
                  
                  // update the list with random input
                  list->editRootMatrix(inputDataMatrixes[order[i]]);
                  list->calculateAndUpdateAll();
                  
                  // backpropagation starts at root
                  BasicLayer* currentLayer = list->getLast();

                  // do math for deltaOutput
                  Matrix3D* currentLayerMatrix = currentLayer->getLayerMatrix();
                  Matrix3D* error = *(outputDataMatrixes[order[i]]) - currentLayerMatrix;
                  Matrix3D* dSig = dSigmoid (currentLayerMatrix);
                  Matrix3D* deltaNext = *error * (dSig);
                  Matrix3D* deltaPrev = new Matrix3D (deltaNext->getLength(), deltaNext->getWidth(), deltaNext->getHeight());

                  delete error;
                  delete dSig;

                  deltaPrev->setMatrix(deltaNext);

                  // calculate and set the bias
                  currentLayer = currentLayer->getPrev();
                  
                  int counter = 0;
                  // list->print(true, true);
                  while (currentLayer->getPrev() != nullptr) {

                     // counter for debuggin
                     counter++;

                     // get the layerMatrix
                     currentLayerMatrix = currentLayer->getLayerMatrix();

							int currentLength = currentLayerMatrix->getLength();
							int currentWidth = currentLayerMatrix->getWidth();
							int currentHeight = currentLayerMatrix->getHeight();
                     // currentLayerMatrix->printMatrix();
                     delete deltaPrev;
                     deltaPrev = new Matrix3D (deltaNext->getLength(), deltaNext->getWidth(), deltaNext->getHeight());
                     deltaPrev->setMatrix(deltaNext);
                   
							error = new Matrix3D (currentLength, currentWidth, currentHeight);

							// calculates the impact of each node and puts it into the weighted matrix. This is used to calculate error
                     for (int l = 0; l < currentLength; l++) {
                        for (int w = 0; w < currentWidth; w++) {
                           for (int h = 0; h < currentHeight; h++) {
               
										// calculates the error for a single node
										float sum = sumMultiplyOfMatrixesHost (currentLayer->getWeights(l,w,h), deltaPrev);
										// inserts the error into a matrix
                              error->insert(sum, l, w, h);
                           }
                        }
                     }
                     // }

                     delete deltaNext;
                     dSig = dSigmoid (currentLayerMatrix);
                     deltaNext = *error * (dSig);
                     delete error;
                     delete dSig;

                     // <--> //

                     // calculate the bias for this node
                     Matrix3D* bias = *deltaPrev * learningRate; 

                     currentLayer->setBiasMatrix(bias);
                     delete bias;

                     // <--> //


                     //calculate the weights for this node
                     for (int l = 0; l < currentLength; l++) {
                        for (int w = 0; w < currentWidth; w++) {
                           for (int h = 0; h < currentHeight; h++) {
                              // up to here gets each node in the matrix
                              float* nodeValue = currentLayerMatrix->getData(l, w, h);
                              
                              Matrix3D* weightMatrix = currentLayer->getWeights(l, w, h);

                                       // weightMatrix->printMatrix();
                                       // deltaPrev->printMatrix();
                              float value = 0;
                              
                              // std::cout << l << " " << w << " " << h << "\n";
                              for (int l2 = 0; l2 < weightMatrix->getLength(); l2++) {
                                 for (int w2 = 0; w2 < weightMatrix->getWidth(); w2++) {
                                    for (int h2 = 0; h2 < weightMatrix->getHeight(); h2++) {
                                       // up to here gets each weight in each node
                                       // weight = 
                                       // std::cout << l2 << " " << w2 << " " << h2 << " " << value << "\n";
                                       // std::cout << *weightMatrix->getData(l2, w2, h2) <<  " " <<*nodeValue <<  " " << *deltaPrev->getData(l2, w2, h2) <<  " " << learningRate << '\n';
                                      
                                       value = *weightMatrix->getData(l2, w2, h2) + *nodeValue * *deltaPrev->getData(l2, w2, h2) * learningRate;
                                       weightMatrix->insert(value, l2, w2, h2);
                                    }
                                 }
                              }
                           }
                        }
                     }

                     currentLayer = currentLayer->getPrev();
                  }

                  delete deltaNext;
                  delete deltaPrev;
               }

               // list->getLast()->print();
               if (isnan (*list->getLast()->getLayerMatrix()->getData(0,0,0))) {
                  std::cout << "here2";
                  exit (0);
               }
            }
            struct rusage usage;
            getrusage (RUSAGE_SELF, &usage);
            std::cout << "\nMemory used (MB): " << usage.ru_maxrss / 1000000 << "\n\n";

            if (print) {
               list->print(true, true);
            }
            // outputDataMatrixes[0]->printMatrix();

            // output results
            if (print) {
               for (int i = 0; i < inputCount; i++) {
                  std::cout << "\n\n\n\n";
                  std::cout << "Input Matrix: ";
                  inputDataMatrixes[i]->printMatrix();
                  std::cout << "True Output: ";
                  outputDataMatrixes[i]->printMatrix();
                  list->editRootMatrix(inputDataMatrixes[i]);

                  // std::cout << "before";
                  // list->print(true, true);
                  list->calculateAndUpdateAll();
                  // std::cout << "after";
                  // list->print();
                  std::cout << "Calculated Output: ";
                  list->getLast()->getLayerMatrix()->printMatrix();
                  std::cout << "\n\n";
               }
            }

            // final error
            std::cout.precision(4);
            if (calculateError) {
               double sumFinal = 0;
               for (int i = 0; i < inputCount; i++) {
                  list->editRootMatrix(inputDataMatrixes[i]);
                  list->calculateAndUpdateAll();
                  Matrix3D* error = *outputDataMatrixes[i] - list->getLast()->getLayerMatrix();
                  Matrix3D* squared = *error * error;
                  sumFinal += squared->sum() * 100;
                  delete error;
                  delete squared;
               }
               std::cout << "Total initial error :: " << sumInitial << "%\n";
               std::cout << "Total final error :: " << sumFinal << "%\n";
            }

            delete[] order;
         }
      }
   }
}