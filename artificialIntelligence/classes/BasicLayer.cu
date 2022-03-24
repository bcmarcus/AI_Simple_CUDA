#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <coreutils/util/time.hpp>

#include <coreutils/util/cudaErrors.cuh>
#include <coreutils/functions/debug/print.cpp>

#include "../functions/activationFunctions.cuh"
#include "BasicLayer.cuh"
#include "BasicWeight.cuh"

using namespace std;
using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::debug;
using namespace artificialIntelligence::classes;
using namespace artificialIntelligence::functions::activation;


// a weight list of lists of lists of Matrix3D
#define MAX_BLOCK_SIZE 8192

BasicLayer::BasicLayer (Matrix3D* layerMatrix, Matrix3D* biasMatrix, BasicWeight* weights) {
   this->layerMatrix = new Matrix3D(layerMatrix->getLength(), layerMatrix->getWidth(), layerMatrix->getHeight());
   this->layerMatrix->setMatrix(layerMatrix);
   if (biasMatrix != nullptr) {
      this->biasMatrix = new Matrix3D(biasMatrix->getLength(), biasMatrix->getWidth(), biasMatrix->getHeight());
      this->biasMatrix->setMatrix(biasMatrix);
   } else {
      this->biasMatrix = nullptr;
   }
   this->weights = weights;
   this->next = nullptr;
   this->prev = nullptr;
}


BasicLayer::BasicLayer (int length, int width, int height) {
   this->layerMatrix = new Matrix3D (length, width, height);
   this->layerMatrix->randomize();
   this->biasMatrix = nullptr;
   this->weights = nullptr;
   this->next = nullptr;
   this->prev = nullptr;
}


BasicLayer::BasicLayer () {
   this->layerMatrix = nullptr;
   this->biasMatrix = nullptr;
   this->weights = nullptr;
   this->next = nullptr;
   this->prev = nullptr;
}


BasicLayer::~BasicLayer () { 
   if (this->layerMatrix != nullptr) {
      delete this->layerMatrix;
   }
   if (this->biasMatrix != nullptr) {
      delete this->biasMatrix;
   }
   if (this->weights != nullptr) {
      delete this->weights;
   }
   if (this->next != nullptr) {
		delete this->next;
	}
}

int BasicLayer::print (bool printBias, bool printWeights, int depth) {
   if (this->layerMatrix != nullptr) {
      std::cout << "\n\nCurrent Index: " << depth << '\n';
      std::cout << "Layer Matrix: \n";
      this->layerMatrix->printMatrix();
   } else {
      std::cout << "No layer found!\n";
      return depth;
   }
   if (printBias) {
      if (this->biasMatrix != nullptr) {
         std::cout << "Bias Matrix: \n";
         this->biasMatrix->printMatrix();
      } else {
         std::cout << "No biases found!\n";
      }
   }
   if (printWeights) {
      if (this->weights != nullptr) {
			std::cout << "Weight Matrix: \n";
         this->weights->print();
      } else {
         std::cout << "No weights found!\n";
      }
   }
   if (this->next == nullptr) {
      return depth;
   }
   return this->next->print(printBias, printWeights, depth + 1);;
}  

BasicLayer* BasicLayer::add (Matrix3D* layerMatrix, Matrix3D* biasMatrix, BasicWeight* weights) {
	
   if (next == nullptr) {
      this->next = new BasicLayer (layerMatrix, nullptr, nullptr);
      this->next->setPrev(this);
      if (this->biasMatrix == nullptr) {
         // std::cout << this->next->layerMatrix->getLength() << " " << this->next->layerMatrix->getWidth() << " " << this->next->layerMatrix->getHeight();
         this->biasMatrix = new Matrix3D(this->next->layerMatrix->getLength(), this->next->layerMatrix->getWidth(), this->next->layerMatrix->getHeight());
         this->biasMatrix->randomize(-0.05, 0.05);
      } else {
         this->biasMatrix->setMatrix(biasMatrix);
      }
		
		this->weights = this->newWeight(this, this->next);
      return this;
   }
   this->next->add(layerMatrix, biasMatrix, weights);
   return this;
}

BasicWeight* BasicLayer::newWeight(BasicLayer* firstLayer, BasicLayer* secondLayer) {
	return new BasicWeight (
		firstLayer->getLayerMatrix()->getLength(), 
		firstLayer->getLayerMatrix()->getWidth(),
		firstLayer->getLayerMatrix()->getHeight(),
		secondLayer->getLayerMatrix()->getLength(),
		secondLayer->getLayerMatrix()->getWidth(),
		secondLayer->getLayerMatrix()->getHeight());
}


BasicLayer* BasicLayer::add (BasicLayer* layer) {
   if (this->next == nullptr) {
      this->next = layer;
   } else {
      this->next = this->next->add(layer);
   }
   return this;
}

void BasicLayer::calculateAndUpdateAllGPUV2() {
	BasicLayer* currentLayer = this;
	Matrix3D* currentLayerMatrix = currentLayer->getLayerMatrix();

	long long numInputs = currentLayerMatrix->getSize() / sizeof(float); // the number of blocks that will be generated
	long long numOutputs = currentLayer->getNext()->getLayerMatrix()->getSize() / sizeof(float);
	long long numWeights = numInputs * numOutputs;
	long long numOutputsRemaining = numOutputs;
	long long outputIndex = 0;

	long long numBlocks = numOutputs > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : numOutputs; // break the number of blocks into chunks of 16384 or less
	long long numThreads = 512; // arbitrary
	long long maxWeightIndex = currentLayer->getWeights()->getWeightMatrix(0)->getSize() / sizeof(float);
	long long numPerThread = std::ceil ((double)maxWeightIndex / (numBlocks * numThreads)); // number of weights per iteration
	long long sharedSize = numThreads * sizeof(float); 
	if (maxWeightIndex > numWeights) {
		maxWeightIndex = numWeights;
	}

	float* input = currentLayerMatrix->getArr();
	float* output = currentLayer->getNext()->getLayerMatrix()->getArr();
	float* current_input;
	float* current_output;
	gpuErrchk(cudaMalloc((void **) &current_input, currentLayerMatrix->getSize()));
	gpuErrchk(cudaMalloc((void **) &current_output, numOutputs * sizeof(float)));

	// streams for asynchronous
	cudaStream_t stream1, stream2;
	cudaStreamCreate ( &stream1); 
	cudaStreamCreate ( &stream2); 
	
	BasicWeight* currentWeight = currentLayer->getWeights();
	long long currentWeightMatrixIndex = 0;
	long long weightsAddedLastSet = 0;
	long long weightsInCurrentKernelRun = 0;

	// std::cout << "Number of threads: " << numThreads << '\n';
	// std::cout << "Number of blocks: " << numBlocks << '\n';
	// std::cout << "Number per thread: " << numPerThread << '\n';
	// std::cout << "Number of bytes for shared storage: " << sharedSize << "\n";
	// std::cout << "Max array index: " << maxWeightIndex << "\n";
	// std::cout << "Max byte index: " << maxWeightIndex * sizeof(float) << "\n";

	float* current_weights;
	float* next_weights;

	gpuErrchk(cudaMalloc((void **) &current_weights, maxWeightIndex * sizeof(float)));
	gpuErrchk(cudaMalloc((void **) &next_weights, maxWeightIndex * sizeof(float)));
	gpuErrchk(cudaMemcpy(current_weights, currentWeight->getWeightMatrix(0)->getArr(), maxWeightIndex * sizeof(float), cudaMemcpyHostToDevice));
	weightsInCurrentKernelRun = maxWeightIndex;
	weightsAddedLastSet = maxWeightIndex;
	currentWeightMatrixIndex++;
	
	int startingOutputID = 0;
	int nextOutputID = maxWeightIndex % currentWeight->outputSize;
	int numWeightsMatrixesLeft = std::ceil((float)numWeights / maxWeightIndex) - 1;

	gpuErrchk(cudaMemcpy(current_input, input, currentLayerMatrix->getSize(), cudaMemcpyHostToDevice)); // set the input layer to the input
	gpuErrchk(cudaMemset(current_output, 0b00000000, numOutputs * sizeof(float))); // set the output to zero

	int debugCounter = 0;
	// go through every single layer

	numWeightsMatrixesLeft = std::ceil((float)numWeights / maxWeightIndex) - 1;
   while (currentLayer->getNext() != nullptr) {
		currentWeightMatrixIndex = 1;
		outputIndex = 0;
		startingOutputID = 0;
		numOutputsRemaining = numOutputs;
		nextOutputID = weightsAddedLastSet;
		int weightsAdded = 0;
		
		bool weightsFinished = false;
		long long weightsUsed = 0;
		do {
			
			// keep adding weights for this layer
			if (numWeightsMatrixesLeft >= 1){
				if (currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float) < maxWeightIndex) {
					maxWeightIndex = currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float);
				}
				gpuErrchk(cudaMemcpyAsync(next_weights, currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr(), maxWeightIndex * sizeof(float), cudaMemcpyHostToDevice));
				// std::cout << "currentWeightMatrixIndex: " << currentWeightMatrixIndex << '\n';
				// std::cout << "maxWeightIndex: " << maxWeightIndex << '\n';
				// std::cout << "currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr() " << currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr()[maxWeightIndex - 1] << "\n\n";
				weightsAddedLastSet = maxWeightIndex;
				currentWeightMatrixIndex++;
				numWeightsMatrixesLeft -= 1;
			} 
			
			// start working on the next layer
			else { 
				if (currentLayer->getNext()->getNext() != nullptr) {
					int nextNumWeights = numOutputs * currentLayer->getNext()->getNext()->getLayerMatrix()->getSize() / sizeof(float);
					int nextMaxWeightIndex = currentLayer->getNext()->getWeights()->getWeightMatrix(0)->getSize() / sizeof(float);
					if (nextMaxWeightIndex > nextNumWeights) {
						nextMaxWeightIndex = nextNumWeights;
					}
					
					gpuErrchk(cudaFree(next_weights));
					gpuErrchk(cudaMalloc((void **) &next_weights, nextMaxWeightIndex * sizeof(float)));
					gpuErrchk(cudaMemcpyAsync(next_weights, currentLayer->getNext()->getWeights()->getWeightMatrix(0)->getArr(), nextMaxWeightIndex * sizeof(float), cudaMemcpyHostToDevice));
					// std::cout << "currentLayer->getNext()->getWeights()->getWeightMatrix(0)->getArr() " << currentLayer->getNext()->getWeights()->getWeightMatrix(0)->getArr()[nextMaxWeightIndex - 1] << "\n\n";
					currentWeightMatrixIndex = 1;
					numWeightsMatrixesLeft = std::ceil((float)nextNumWeights / nextMaxWeightIndex) - 1;
					// std::cout << "numWeightsMatrixesLeft: " << numWeightsMatrixesLeft << "\n";
					weightsAddedLastSet = nextMaxWeightIndex;
				}
				weightsFinished = true;
			}
			
			long long helper = 0;

			// works
			// run through all of the outputs and multiply the inputs by the weights to get the actual output value
			do {
				if (numOutputsRemaining > 0) {
					// std::cout << "inside22\n";
					// std::cout << "numBlocks: " << numBlocks << '\n';
					// std::cout << "numOutputs: " << numOutputs << '\n';
					// std::cout << "numPerThread: " << numPerThread << '\n';
					// std::cout << "weightsInCurrentKernelRun: " << weightsInCurrentKernelRun << "\n";
					// std::cout << "numOutputsRemaining: " << numOutputsRemaining << '\n';
					// std::cout << "helper: " << helper << '\n';
					// std::cout << "weightsUsed: " << weightsUsed << "\n";
					// std::cout << "numWeightsMatrixesLeft: " << numWeightsMatrixesLeft << "\n";
					// std::cout << "weightsAddedLastSet: " << weightsAddedLastSet << "\n";
					// std::cout << "startingOutputID: " << startingOutputID << "\n\n";
					
					if (numOutputsRemaining - numBlocks < 0) {
						numBlocks = numOutputsRemaining;
					}

					artificialIntelligence::classes::calculateAndUpdateLayerGPU<<< numBlocks, numThreads, sharedSize, stream1 >>>(current_input, current_weights, current_output, numBlocks, numOutputs, numPerThread, weightsInCurrentKernelRun, helper, weightsUsed, startingOutputID);
					outputIndex += numBlocks;
					numOutputsRemaining -= numBlocks;
				}
				startingOutputID += numBlocks;
				helper += numBlocks;

			} while (numOutputsRemaining > 0);
			gpuErrchk(cudaDeviceSynchronize());
			
			startingOutputID = nextOutputID % numOutputs;
			nextOutputID += weightsInCurrentKernelRun % numOutputs;
			numOutputsRemaining = numOutputs;

			weightsUsed += weightsInCurrentKernelRun;
			numBlocks = numOutputs > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : numOutputs;
			weightsInCurrentKernelRun = weightsAddedLastSet;

			float* temp = current_weights;
			current_weights = next_weights;
			next_weights = temp;

		} while (!weightsFinished);

		gpuErrchk(cudaMemcpy(output, current_output, numOutputs * sizeof(float), cudaMemcpyDeviceToHost));
		
		Matrix3D* bias = currentLayer->getBias();
		currentLayer = currentLayer->getNext();
		currentLayerMatrix = currentLayer->getLayerMatrix();
		currentWeight = currentLayer->getWeights();
		numInputs = currentLayerMatrix->getSize() / sizeof(float);

		if (currentLayer->getNext() != nullptr) {
			output = currentLayer->getNext()->getLayerMatrix()->getArr();
			numOutputs = currentLayer->getNext()->getLayerMatrix()->getSize() / sizeof(float);
			numWeights = numInputs * numOutputs;
			maxWeightIndex = currentLayer->weights->getWeightMatrix(0)->getSize();
			numBlocks = numOutputs > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : numOutputs; // break the number of blocks into chunks of 16384 or less
			numThreads = 512; // arbitrary
			numPerThread = std::ceil ((double)maxWeightIndex / (numBlocks * numThreads));
			output = currentLayer->getNext()->getLayerMatrix()->getArr();
			gpuErrchk(cudaFree(next_weights));
			gpuErrchk(cudaMalloc((void **) &next_weights, maxWeightIndex * sizeof(float)));
			gpuErrchk(cudaFree(current_output));
			gpuErrchk(cudaMalloc((void **) &current_output, numOutputs * sizeof(float)));
			gpuErrchk(cudaMemset(current_output, 0b00000000, numOutputs * sizeof(float))); // set the output to zero
		}
		
		*currentLayer->getLayerMatrix() += bias;
		sigmoid(currentLayer->getLayerMatrix(), false);
		gpuErrchk(cudaFree(current_input));
		gpuErrchk(cudaMalloc((void **) &current_input, currentLayerMatrix->getSize()));
		input = currentLayerMatrix->getArr();
		gpuErrchk(cudaMemcpy(current_input, input, currentLayerMatrix->getSize(), cudaMemcpyHostToDevice)); // set the input layer to the input

		debugCounter++;
	}




	// :::: FREE ALL ALLOCATED MEMORY :::: //
	gpuErrchk(cudaFree(current_input));	
	gpuErrchk(cudaFree(current_output));
	gpuErrchk(cudaFree(current_weights));	
	gpuErrchk(cudaFree(next_weights));

}

// works as intended for small sets at least
__global__ void artificialIntelligence::classes::calculateAndUpdateLayerGPU(float* nodeValues, float* weights, float* output, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned long long outputNodeId = (blockIdx.x + startingOutputId) % outputSize;
	unsigned int numThreads = blockDim.x;
	unsigned long long weightIndex = tid * outputSize + blockIdx.x + helperIndex;
	unsigned long long inputNodeId = 0;
	unsigned int gridSize = numThreads*outputSize;
	sdata[tid] = 0;
	// printf("startingOutputId: %d\n", startingOutputId);

	while (weightIndex < maxWeightIndex) {
		inputNodeId = (weightIndex + startingWeight) / outputSize;
		sdata[tid] += nodeValues[inputNodeId] * weights[weightIndex];
		// if (weightIndex < 20 || weightIndex > 8150 && weightIndex < 8200) printf("NodeId: %d  Node: %f  outputNodeId: %d   WeightId: %d  Weight: %f   Data: %f\n", outputSize, nodeValues[inputNodeId], outputNodeId, weightIndex, weights[weightIndex], sdata[tid]);
		weightIndex += gridSize;
	}

	__syncthreads();

	for (unsigned int s=numThreads/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		// if (outputNodeId < 20) {printf ("BlockID: %d   Data:  %f   Output: %f\n", outputNodeId, sdata[0], output[outputNodeId] + sdata[0]);}
		output[outputNodeId] += sdata[0];
	}
}

void artificialIntelligence::classes::BasicLayer::calculateAndUpdateAllCPU () {
   if (this->next == nullptr) {
      return;
   }
   this->calculateAndUpdateLayerCPU();
   this->next->calculateAndUpdateAllCPU();
}

// can be parallelized
void BasicLayer::calculateAndUpdateLayerCPU () {
   // start with the first node, and add all of the values to a node then sigmoid

   Matrix3D* nextLayer = this->next->getLayerMatrix();
   Matrix3D* outputs = new Matrix3D (nextLayer->getLength(), nextLayer->getWidth(), nextLayer->getHeight());
	outputs->setAll(0);
   if (isnan(*outputs->getData(0, 0, 0))) {
      std::cout << "null init";
      exit (0);
   }
   // start at start layer, then go to the end layer
   
   // declaring temp variables
   float activation = 0;

   // loop through every weight matrix
   // std::cout << "[" << this->layerMatrix->getLength() << "] " << "[" << this->layerMatrix->getWidth() << "] " << "[" << this->layerMatrix->getHeight() << "]   " 
   // << "[" << nextLayer->getLength() << "] " << "[" << nextLayer->getWidth() << "] " << "[" << nextLayer->getHeight() << "]" << "\n\n";
   for (int fl = 0; fl < this->layerMatrix->getLength(); fl++) {
      for (int fw = 0; fw < this->layerMatrix->getWidth(); fw++) {
         for (int fh = 0; fh < this->layerMatrix->getHeight(); fh++) {
            

            for (int sl = 0; sl < nextLayer->getLength(); sl++) {
               for (int sw = 0; sw < nextLayer->getWidth(); sw++) {
                  for (int sh = 0; sh < nextLayer->getHeight(); sh++) {
                     

                     
                     activation = *this->layerMatrix->getData(fl, fw, fh) * *this->weights->getData(fl, fw, fh, sl, sw, sh) + *outputs->getData(sl, sw, sh);

                     outputs->insert(activation, sl, sw, sh);
                  }
               }
            }
         }
      }
   } 

   // adds the bias and takes the sigmoid
   for (int sl = 0; sl < nextLayer->getLength(); sl++) {
      for (int sw = 0; sw < nextLayer->getWidth(); sw++) {
         for (int sh = 0; sh < nextLayer->getHeight(); sh++) {
            activation = sigmoid(*outputs->getData(sl, sw, sh) + *this->biasMatrix->getData(sl, sw, sh));
            // std::cout << outputs->getData(sl, sw, sh) << '\n';
            outputs->insert(activation, sl, sw, sh);
         }
      }
   }

   // set the next matrix to the layer that was just found
   this->next->setLayerMatrix (outputs);
   delete outputs;

}


void BasicLayer::setPrev (BasicLayer* prev) {
   if (this->prev != nullptr) {
      delete this->prev;
   }
   this->prev = prev;
}


Matrix3D* BasicLayer::getLayerMatrix () {
   return this->layerMatrix;
}


void BasicLayer::setLayerMatrix (Matrix3D* layerMatrix) {
   if (this->layerMatrix == nullptr) {
      this->layerMatrix = new Matrix3D (layerMatrix->getLength(), layerMatrix->getWidth(), layerMatrix->getHeight());
   }
   this->layerMatrix->setMatrix(layerMatrix);
}


BasicWeight* BasicLayer::getWeights () {
	return this->weights;
}


Matrix3D* BasicLayer::getBias () {
   return this->biasMatrix;
}


void BasicLayer::setBiasMatrix (Matrix3D* biasMatrix) {
   if (this->biasMatrix == nullptr) {
      this->biasMatrix = new Matrix3D (biasMatrix->getLength(), biasMatrix->getWidth(), biasMatrix->getHeight());
   }
   this->biasMatrix->setMatrix(biasMatrix);
}


BasicLayer* BasicLayer::getLast () {
   if (this->next == nullptr) {
      return this;
   }
   return this->next->getLast();
}


BasicLayer* BasicLayer::getNext () {
   return this->next;
}


BasicLayer* BasicLayer::getPrev () {
   return this->prev;
}

Matrix3D* BasicLayer::calculateErrorCPU (Matrix3D* delta) {
	Matrix3D* currentLayerMatrix = this->layerMatrix;
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	for (int l = 0; l < currentLayerMatrix->getLength(); l++) {
		for (int w = 0; w < currentLayerMatrix->getWidth(); w++) {
			for (int h = 0; h < currentLayerMatrix->getHeight(); h++) {
				// currentLayer->print(true, true);
				Matrix3D* outputMatrix = this->getNext()->getLayerMatrix();
				Matrix3D* weightedMatrix = new Matrix3D (delta->getLength(), delta->getWidth(), delta->getHeight());
				//*model->getRoot()->getWeights(l, w, h) * deltaPrev;
				for (int l2 = 0; l2 < outputMatrix->getLength(); l2++) {
					for (int w2 = 0; w2 < outputMatrix->getWidth(); w2++) {
						for (int h2 = 0; h2 < outputMatrix->getHeight(); h2++) {
							// std::cout << "*currentLayer->getWeights()->getData(l, w, h, l2, w2, h2): " << *currentLayer->getWeights()->getData(l, w, h, l2, w2, h2) << '\n';
							// std::cout << "*deltaPrev->getData(l2, w2, h2): " << *deltaPrev->getData(l2, w2, h2) << '\n';
							weightedMatrix->insert(*this->getWeights()->getData(l, w, h, l2, w2, h2) * *delta->getData(l2, w2, h2), l2, w2, h2);
						}
					}
				}
				error->insert(weightedMatrix->sum(), l, w, h);
				delete weightedMatrix;
			}
		}
	}
	return error;
}

Matrix3D* BasicLayer::calculateErrorGPU (Matrix3D* delta) {
	BasicLayer* currentLayer = this;
	Matrix3D* currentLayerMatrix = currentLayer->getLayerMatrix();

	long long numInputs = currentLayerMatrix->getSize() / sizeof(float); // the number of blocks that will be generated
	long long numOutputs = currentLayer->getNext()->getLayerMatrix()->getSize() / sizeof(float);
	long long numWeights = numInputs * numOutputs;
	long long numInputsRemaining = numInputs;
	long long numWeightsRemaining = numWeights;
	long long inputIndex = 0;
	long long numBlocks = numInputs > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : numInputs; 
	long long numThreads = 512; // arbitrary
	long long maxWeightIndex = numBlocks * numOutputs;
	long long numPerThread = std::ceil ((double)maxWeightIndex / (numBlocks * numThreads)); // number of weights per iteration
	long long sharedSize = numThreads * sizeof(float); 
	if (maxWeightIndex > numWeights) {
		maxWeightIndex = numWeights;
	}
	Matrix3D* errorMatrix = new Matrix3D(currentLayer->getLayerMatrix()->getLength(), currentLayer->getLayerMatrix()->getWidth(), currentLayer->getLayerMatrix()->getHeight());
	float* error = errorMatrix->getArr();
	float* current_error;
	float* current_delta;
	gpuErrchk(cudaMalloc((void **) &current_error, errorMatrix->getSize()));
	gpuErrchk(cudaMalloc((void **) &current_delta, delta->getSize()));
	gpuErrchk(cudaMemcpy(current_error, error, errorMatrix->getSize(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(current_delta, delta->getArr(), delta->getSize(), cudaMemcpyHostToDevice));

	// streams for asynchronous
	cudaStream_t stream1, stream2;
	cudaStreamCreate ( &stream1); 
	cudaStreamCreate ( &stream2); 
	
	BasicWeight* currentWeight = currentLayer->getWeights();
	long long matrixSize = currentWeight->weights->getSize() / sizeof(float);
	long long numLeftToAdd = maxWeightIndex;
	long long weightIndex = 0;
	long long numLastAdded = 0;
	long long currentWeightMatrixIndex = 0;
	long long weightsAddedLastSet = 0;
	long long weightsInCurrentKernelRun = 0;

	// std::cout << "Number of threads: " << numThreads << '\n';
	// std::cout << "Number of blocks: " << numBlocks << '\n';
	// std::cout << "Number per thread: " << numPerThread << '\n';
	// std::cout << "Number of bytes for shared storage: " << sharedSize << "\n";
	// std::cout << "Max array index: " << maxWeightIndex << "\n";
	// std::cout << "Max byte index: " << maxWeightIndex * sizeof(float) << "\n";
	// std::cout << "numLeftToAdd: " <<  numLeftToAdd << "\n";

	float* current_weights;
	float* next_weights;

	gpuErrchk(cudaMalloc((void **) &current_weights, maxWeightIndex * sizeof(float)));
	gpuErrchk(cudaMalloc((void **) &next_weights, maxWeightIndex * sizeof(float)));

	// copy the number of weights the corresponds to the number of weights based off of the inputs
	int weightsInCurrentMatrix = currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float);
	int weightsInBasicWeight = currentWeight->getSize();

	int numberOfWeightsToAdd = numBlocks * numOutputs;
	int toAdd = weightsInCurrentMatrix > numberOfWeightsToAdd ? numberOfWeightsToAdd : weightsInCurrentMatrix;

	int amountAdded = 0;
	int weightsAdded = 0;
	int currentWeightIndex = 0;

	while (numberOfWeightsToAdd > 0) {
		toAdd = weightsInCurrentMatrix > numberOfWeightsToAdd ? numberOfWeightsToAdd : weightsInCurrentMatrix;
		// std::cout << "\ntoAdd: " <<  toAdd << "\n";
		// std::cout << "numberOfWeightsToAdd: " <<  numberOfWeightsToAdd << "\n";
		// std::cout << "weightsInCurrentMatrix: " <<  weightsInCurrentMatrix << "\n";
		// std::cout << "currentWeightMatrixIndex: " <<  currentWeightMatrixIndex << "\n";
		gpuErrchk(cudaMemcpy(&current_weights[weightsAdded], &currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr()[amountAdded], toAdd * sizeof(float), cudaMemcpyHostToDevice));
		if (toAdd == weightsInCurrentMatrix) {
			currentWeightMatrixIndex++;
			// std::cout << "inside1\n";
			numberOfWeightsToAdd -= toAdd;
			amountAdded = 0;
			weightsAdded += toAdd;
			if (weightsAdded < numWeights) {
				weightsInCurrentMatrix = currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float);
			}
		} else {
			amountAdded = toAdd;
			numberOfWeightsToAdd = 0;
			weightsInCurrentMatrix -= toAdd;
			weightsAdded += toAdd;
			// std::cout << "weightsAdded: " << weightsAdded << '\n';
		}
	}
	// amountAdded = 0;
	weightsInCurrentKernelRun = weightsAdded;
	weightsAddedLastSet = weightsAdded;
	
	int startingInputID = 0;
	int nextInputID = maxWeightIndex % currentWeight->outputSize;
	int numWeightsMatrixesLeft = std::ceil((float)numWeights / maxWeightIndex) - 1;
	int debugCounter = 0;
	// go through every single layer
	numWeightsMatrixesLeft = std::ceil((float)numWeights / maxWeightIndex) - 1;
	inputIndex = 0;
	startingInputID = 0;
	numInputsRemaining = numInputs;
	nextInputID = weightsAddedLastSet;
	bool weightsFinished = false;
	long long weightsUsed = 0;
	do {

		// run the kernel with the first set of inputs and weights
		if (numInputsRemaining > 0) {
			numBlocks = (weightsUsed + weightsInCurrentKernelRun) * numInputs / numWeights - weightsUsed * numInputs / numWeights;
			// std::cout << "inside22\n";
			// std::cout << "numBlocks: " << numBlocks << '\n';
			// std::cout << "numInputs: " << numInputs << '\n';
			// std::cout << "numPerThread: " << numPerThread << '\n';
			// std::cout << "weightsInCurrentKernelRun: " << weightsInCurrentKernelRun << "\n";
			// std::cout << "numInputsRemaining: " << numInputsRemaining << '\n';
			// std::cout << "weightsUsed: " << weightsUsed << "\n";
			// std::cout << "startingInputID: " << startingInputID << "\n\n";

			if (numInputsRemaining - numBlocks < 0) {
				numBlocks = numInputsRemaining;
			}

			// needs to make it so that the first x inputs are all added together by the blocks that exist. 
			
			artificialIntelligence::classes::calculateError<<< numBlocks, numThreads, sharedSize, stream1 >>>(current_weights, current_delta, current_error, numInputs, numOutputs, numPerThread, weightsInCurrentKernelRun, numWeights, weightsUsed, startingInputID);
			inputIndex += numBlocks;
			numInputsRemaining -= numBlocks;
		}
		gpuErrchk(cudaDeviceSynchronize());
		startingInputID += numBlocks;

		weightsUsed += weightsInCurrentKernelRun;
		weightsInCurrentKernelRun = weightsAddedLastSet;
		if (numWeights - weightsAdded > 0) {
			// std::cout << "here\n";
			// exit(0);
			// asynchronously add the next set of weights
			numBlocks = numInputsRemaining > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : numInputsRemaining;
			numberOfWeightsToAdd = numBlocks * numOutputs;
			toAdd = weightsInCurrentMatrix > numberOfWeightsToAdd ? numberOfWeightsToAdd : weightsInCurrentMatrix;
			amountAdded = weightsAdded % (BASIC_WEIGHT_MAX_SIZE);
			int weightCounter = 0;
			if (weightsInCurrentMatrix > 0) {
				weightsInCurrentKernelRun = numberOfWeightsToAdd;
				while (numberOfWeightsToAdd > 0) {
					toAdd = weightsInCurrentMatrix > numberOfWeightsToAdd ? numberOfWeightsToAdd : weightsInCurrentMatrix;
					// std::cout << "currentWeightMatrixIndex: " << currentWeightMatrixIndex << "\n";
					// std::cout << "amountAdded: " << amountAdded << "\n";
					// std::cout << "toAdd: " << toAdd << "\n";
					// std::cout << "weightsInCurrentMatrix: " << weightsInCurrentMatrix << "\n";
					// std::cout << "currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr()[amountAdded]: " << currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr()[amountAdded] << '\n';
					gpuErrchk(cudaMemcpyAsync(&next_weights[weightCounter], &currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr()[amountAdded], toAdd * sizeof(float), cudaMemcpyHostToDevice));
					if (toAdd == weightsInCurrentMatrix) {
						currentWeightMatrixIndex++;
						numberOfWeightsToAdd -= toAdd;
						amountAdded = 0;
						weightsAdded += toAdd;
						weightCounter += toAdd;
						if (weightsAdded < numWeights) {
							weightsInCurrentMatrix = currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float);
						}
					} else {
						numberOfWeightsToAdd = 0;
						weightsInCurrentMatrix -= toAdd;
						weightsAdded += toAdd;
					}
				}
			}
		}
		else {
			weightsFinished = true;
		}

		gpuErrchk(cudaDeviceSynchronize());

		float* temp = current_weights;
		current_weights = next_weights;
		next_weights = temp;

	} while (!weightsFinished);

	gpuErrchk(cudaMemcpy(error, current_error, numInputs * sizeof(float), cudaMemcpyDeviceToHost));

	// :::: FREE ALL ALLOCATED MEMORY :::: //
	gpuErrchk(cudaFree(current_error));	
	gpuErrchk(cudaFree(current_delta));
	gpuErrchk(cudaFree(current_weights));	
	gpuErrchk(cudaFree(next_weights));
	return errorMatrix;
}

__global__ void artificialIntelligence::classes::calculateError(float* weights, float* delta, float* error, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingInputID) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int numThreads = blockDim.x;
	unsigned long long inputNodeId = blockIdx.x + startingInputID;
	unsigned long long weightIndex = tid + blockIdx.x * outputSize;
	unsigned int gridSize = numThreads;
	int weightsToAddStart = outputSize * (blockIdx.x);
	int weightsToAddEnd = outputSize * (blockIdx.x + 1);

	sdata[tid] = 0;
	// printf("startingInputID: %d\n", startingInputID);
	// 	printf("maxWeightIndex: %d \n",  startingWeight);
	// 	printf("startingInputID: %d\n", startingInputID);
	// 	printf("deltaId: %d  delta: %f  inputNodeId: %d  Data: %f  ", (startingWeight + weightIndex) % outputSize, delta[(startingWeight + weightIndex) % outputSize], inputNodeId, sdata[tid]);
	// 	printf("WeightId: %d  Weight: %f\n",  weightIndex, weights[weightIndex]);
	// }
	while (weightIndex >= weightsToAddStart && weightIndex < weightsToAddEnd) {
		sdata[tid] += weights[weightIndex] * delta[(startingWeight + weightIndex) % outputSize];
		// if (weightIndex > 67108860) {
		// 	printf("deltaId: %d  delta: %f  inputNodeId: %d  Data: %f  \n", (startingWeight + weightIndex) % outputSize, delta[(startingWeight + weightIndex) % outputSize], weightIndex, sdata[tid]);
		// 	printf("WeightId: %d  Weight: %f\n",  weightIndex, weights[weightIndex]);
		// }
		weightIndex += gridSize;
	}

	__syncthreads();

	for (unsigned int s=numThreads/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		// printf ("BlockID: %d   Data:  %f   Output: %f\n", inputNodeId, sdata[0], error[inputNodeId] + sdata[0]);
		error[inputNodeId] += sdata[0];
	}
}

Matrix3D* BasicLayer::updateWeightsCPU (Matrix3D* delta, double learningRate) {
	Matrix3D* currentLayerMatrix = this->layerMatrix;
	for (int l = 0; l < currentLayerMatrix->getLength(); l++) {
		for (int w = 0; w < currentLayerMatrix->getWidth(); w++) {
			for (int h = 0; h < currentLayerMatrix->getHeight(); h++) {
				// up to here gets each node in the matrix
				float inputValue = *currentLayerMatrix->getData(l, w, h);
				float value = 0;
				
				Matrix3D* weightMatrix = this->getNext()->getLayerMatrix();
				
				for (int l2 = 0; l2 < weightMatrix->getLength(); l2++) {
					for (int w2 = 0; w2 < weightMatrix->getWidth(); w2++) {
						for (int h2 = 0; h2 < weightMatrix->getHeight(); h2++) {
							value = *this->weights->getData(l, w, h, l2, w2, h2) + inputValue * *delta->getData(l2, w2, h2) * learningRate;
							this->weights->insertData(value, l, w, h, l2, w2, h2);
						}
					}
				}
			}
		}
	}
}

// updates weights one at a time, with each kernel doing maxWeightIndex weights
Matrix3D* BasicLayer::updateWeightsGPU (Matrix3D* delta, double learningRate) {
	BasicLayer* currentLayer = this;
	Matrix3D* currentLayerMatrix = currentLayer->getLayerMatrix();

	long long numInputs = currentLayerMatrix->getSize() / sizeof(float); // the number of blocks that will be generated
	long long numOutputs = currentLayer->getNext()->getLayerMatrix()->getSize() / sizeof(float);
	long long numWeights = numInputs * numOutputs;
	long long numInputsRemaining = numInputs;
	long long numWeightsRemaining = numWeights;
	long long inputIndex = 0;
	long long numBlocks = numInputs > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : numInputs; 
	long long numThreads = 512; // arbitrary
	long long maxWeightIndex = numBlocks * numOutputs;
	long long numPerThread = std::ceil ((double)maxWeightIndex / (numBlocks * numThreads)); // number of weights per iteration
	long long sharedSize = numThreads * sizeof(float); 
	if (maxWeightIndex > numWeights) {
		maxWeightIndex = numWeights;
	}


	// streams for asynchronous
	cudaStream_t stream1, stream2;
	cudaStreamCreate ( &stream1); 
	cudaStreamCreate ( &stream2);

	BasicWeight* currentWeight = currentLayer->getWeights();
	long long matrixSize = currentWeight->weights->getSize() / sizeof(float);
	long long numLeftToAdd = maxWeightIndex;
	long long weightIndex = 0;
	long long numLastAdded = 0;
	long long currentWeightMatrixIndex = 0;
	long long weightsAddedLastSet = 0;
	long long weightsInCurrentKernelRun = 0;

	int numberOfWeightsToAdd = maxWeightIndex;
	int weightsInCurrentMatrix = currentWeight->getWeightMatrix(0)->getSize() / sizeof(float);

	int toAdd = weightsInCurrentMatrix > numberOfWeightsToAdd ? numberOfWeightsToAdd : weightsInCurrentMatrix;
	// std::cout << "\ntoAdd: " <<  toAdd << "\n";
	// std::cout << "numberOfWeightsToAdd: " <<  numberOfWeightsToAdd << "\n";
	// std::cout << "weightsInCurrentMatrix: " <<  weightsInCurrentMatrix << "\n";
	// std::cout << "currentWeightMatrixIndex: " <<  currentWeightMatrixIndex << "\n";


	Matrix3D* inputMatrix = currentLayer->getLayerMatrix();
	float* current_input;
	float* current_delta;
	gpuErrchk(cudaMalloc((void **) &current_input, inputMatrix->getSize()));
	gpuErrchk(cudaMalloc((void **) &current_delta, delta->getSize()));
	gpuErrchk(cudaMemcpy(current_input, inputMatrix->getArr(), inputMatrix->getSize(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(current_delta, delta->getArr(), delta->getSize(), cudaMemcpyHostToDevice));
	

	float* current_weights;
	float* next_weights;
	gpuErrchk(cudaMalloc((void **) &current_weights, currentWeight->getWeightMatrix(0)->getSize() / sizeof(float)));
	gpuErrchk(cudaMemcpy(current_weights, currentWeight->getWeightMatrix(0), currentWeight->getWeightMatrix(0)->getSize() / sizeof(float), cudaMemcpyHostToDevice));

	int numWeightsAdded = 0;
	
	int weightsUsed = 0;
	int startingInputId = 0;
	while (numWeights - weightsUsed > 0){
		// std::cout << "inside22\n";
		// std::cout << "numBlocks: " << numBlocks << '\n';
		// std::cout << "numInputs: " << numInputs << '\n';
		// std::cout << "numPerThread: " << numPerThread << '\n';
		// std::cout << "weightsInCurrentKernelRun: " << weightsInCurrentKernelRun << "\n";
		// std::cout << "numInputsRemaining: " << numInputsRemaining << '\n';
		// std::cout << "weightsUsed: " << weightsUsed << "\n";
		// std::cout << "startingInputID: " << startingInputID << "\n\n";

		if (numInputsRemaining - numBlocks < 0) {
			numBlocks = numInputsRemaining;
		}

		// needs to make it so that the first x inputs are all added together by the blocks that exist. 
		
		artificialIntelligence::classes::updateWeights<<<numBlocks, numThreads, sharedSize, stream1>>>(current_weights, current_delta, current_input, numInputs, numOutputs, numPerThread, weightsInCurrentKernelRun, numWeights, weightsUsed, startingInputId, learningRate);
		inputIndex += numBlocks;
		numInputsRemaining -= numBlocks;

		weightsUsed += currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float);

		// add more weights if they exist
		currentWeightMatrixIndex++;
		if (numWeights - weightsUsed > 0) {
			gpuErrchk(cudaMalloc((void **) &next_weights, currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float)));
			gpuErrchk(cudaMemcpy(next_weights, currentWeight->getWeightMatrix(currentWeightMatrixIndex), currentWeight->getWeightMatrix(0)->getSize() / sizeof(float), cudaMemcpyHostToDevice));
		}

		// bring weights back
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(currentWeight->getWeightMatrix(currentWeightMatrixIndex - 1), current_weights, currentWeight->getWeightMatrix(0)->getSize() / sizeof(float), cudaMemcpyHostToDevice));

		float* temp = current_weights;
		current_weights = next_weights;
		next_weights = temp;
	}
}

__global__ void artificialIntelligence::classes::updateWeights(float* weights, float* delta, float* input, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingInputID, double learningRate) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int numThreads = blockDim.x;
	unsigned long long inputNodeId = blockIdx.x + startingInputID;
	unsigned long long outputNodeId = 0;
	unsigned long long weightIndex = tid + blockIdx.x * outputSize;
	unsigned int gridSize = numThreads;
	int weightsToAddStart = outputSize * (blockIdx.x);
	int weightsToAddEnd = outputSize * (blockIdx.x + 1);

	while (weightIndex >= weightsToAddStart && weightIndex < weightsToAddEnd) {
		weights[weightIndex] += input[inputNodeId] * delta[outputNodeId] * learningRate;
		weightIndex += gridSize;
	}
}
// weight[weightIndex] + currentLayer input[value] * deltaPrev [output value] * learningRate
// insert the weight into the proper place
// return weights



void BasicLayer::toFile (std::ofstream* outputFile) {
   *outputFile << this->layerMatrix->getLength() << ',' << this->layerMatrix->getWidth() << ',' << this->layerMatrix->getHeight() << '\n';

   // print bias values
   if (this->biasMatrix == nullptr) {
      return;
   }
   *outputFile << this->biasMatrix->getLength() << ',' << this->biasMatrix->getWidth() << ',' << this->biasMatrix->getHeight() << '\n';
   for (int i = 0; i < this->biasMatrix->getLength(); i++) {
      for (int j = 0; j < this->biasMatrix->getWidth(); j++) {
         for (int k = 0; k < this->biasMatrix->getHeight(); k++) {
            *outputFile << *this->biasMatrix->getData(i, j, k) << ',';
         }
      }
   }

   outputFile->seekp((int) outputFile->tellp() - 1);
   outputFile->write("\n", 1);

   if (this->weights == nullptr) {
      return;
   }

   // print weight values
   *outputFile << this->layerMatrix->getLength() << ',' << this->layerMatrix->getWidth() << ',' << this->layerMatrix->getHeight() << ',';
   *outputFile << this->biasMatrix->getLength() << ',' << this->biasMatrix->getWidth() << ',' << this->biasMatrix->getHeight() << '\n';
   for (int l = 0; l < this->layerMatrix->getLength(); l++) {
      for (int w = 0; w < this->layerMatrix->getWidth(); w++) {
         for (int h = 0; h < this->layerMatrix->getHeight(); h++) {
            for (int l2 = 0; l2 < this->biasMatrix->getLength(); l2++) {
               for (int w2 = 0; w2 < this->biasMatrix->getWidth(); w2++) {
                  for (int h2 = 0; h2 < this->biasMatrix->getHeight(); h2++) {
                     *outputFile << *this->weights->getData(l, w, h, l2, w2, h2) << ',';
                  }
               }
            }
         }
      }
   }

   outputFile->seekp((int) outputFile->tellp() - 1);
   outputFile->write("\n", 1); 

   if (this->next == nullptr) {
      return;
   }
   this->next->toFile(outputFile);
}


BasicLayer* BasicLayer::loadFromFile (std::ifstream* inputFile, BasicLayer* prev) {
   BasicLayer* layer = new BasicLayer ();
   std::string line;
   getline (*inputFile, line);
   std::stringstream lineStream;
   lineStream << line;
   std::string value;
   getline(lineStream, value, ',');
   int layerLength = stoi(value);
   getline(lineStream, value, ',');
   int layerWidth = stoi(value);
   getline(lineStream, value, ',');
   int layerHeight = stoi(value);
   Matrix3D* layerMatrix = new Matrix3D (layerLength, layerWidth, layerHeight);
   layer->layerMatrix = layerMatrix;
   layer->prev = prev;
      // std::cout << layerLength << " " << layerWidth << " " << layerHeight;

   lineStream.str(std::string());
   lineStream.clear();
   getline (*inputFile, line);
   lineStream << line;

   if (inputFile->eof()) {
      layer->biasMatrix = nullptr;
      layer->weights = nullptr;
      return layer;
   }

   getline(lineStream, value, ',');
   int biasLength = stoi(value);
   getline(lineStream, value, ',');
   int biasWidth = stoi(value);
   getline(lineStream, value, ',');
   int biasHeight = stoi(value);
   Matrix3D* biasMatrix = new Matrix3D (biasLength, biasWidth, biasHeight);
   layer->biasMatrix = biasMatrix;

   lineStream.str(std::string());
   lineStream.clear();
   getline (*inputFile, line);
   lineStream << line;
   for (int i = 0; i < layer->biasMatrix->getLength(); i++) {
      for (int j = 0; j < layer->biasMatrix->getWidth(); j++) {
         for (int k = 0; k < layer->biasMatrix->getHeight(); k++) {
            std::getline(lineStream, value, ',');
            layer->biasMatrix->insert (stod(value), i, j, k);
         }
      }
   }

   getline (*inputFile, line);

   if (inputFile->eof()) {
      layer->weights = nullptr;
      return layer;
   }

   BasicWeight* weights = new BasicWeight (
      layer->layerMatrix->getLength(), 
      layer->layerMatrix->getWidth(), 
      layer->layerMatrix->getHeight(), 
      layer->biasMatrix->getLength(), 
      layer->biasMatrix->getWidth(), 
      layer->biasMatrix->getHeight()
   );

   
   lineStream.str(std::string());
   lineStream.clear();
   getline (*inputFile, line);
   lineStream << line;
   // ./te (0);
   for (int l = 0; l < layerLength; l++) {
      for (int w = 0; w < layerWidth; w++) {
         for (int h = 0; h < layerHeight; h++) {
            for (int l2 = 0; l2 < biasLength; l2++) {
               for (int w2 = 0; w2 < biasWidth; w2++) {
                  for (int h2 = 0; h2 < biasHeight; h2++) {
                     std::getline(lineStream, value, ',');
                     weights->insertData(stod(value), l, w, h, l2, w2, h2);
                  }
               }
            }
         }
      }
   }

   layer->weights = weights;
   layer->next = BasicLayer::loadFromFile (inputFile, layer);

   return layer;
}