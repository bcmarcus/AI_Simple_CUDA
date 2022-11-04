#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <unistd.h>

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <coreutils/functions/debug/print.hpp>
#include <coreutils/util/time.hpp>
#include <coreutils/util/cudaErrors.cuh>

#include "../layers/BasicLayer.cuh"
#include "../weights/BasicWeight.cuh"

using namespace std;
using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::debug;
using namespace artificialIntelligence::classes;
using namespace artificialIntelligence::functions::activation;

#define MAX_BLOCK_SIZE 8192

BasicLayer::BasicLayer (Matrix3D* layerMatrix, Matrix3D* biasMatrix, BasicWeight* weights, ActivationType activationType) {
   this->layerMatrixes = new Matrix3D* [1];
	this->layerMatrixes[0] = layerMatrix;

	this->biasMatrixes = new Matrix3D*[1];
	this->biasMatrixes[0] = nullptr;
   if (biasMatrix != nullptr) {
      this->biasMatrixes[0] = new Matrix3D(biasMatrix->getLength(), biasMatrix->getWidth(), biasMatrix->getHeight());
      this->getBias()->setMatrix(biasMatrix);
   }

	this->weights = (WeightBase**) new BasicWeight*[1];
   this->weights[0] = weights;

   this->next = (LayerBase**) new BasicLayer*[1];
	this->next[0] = nullptr;
   this->prev = (LayerBase**) new BasicLayer*[1];
	this->prev[0] = nullptr;

	this->layerMatrixCount = 1;
	this->biasCount = biasMatrix != nullptr;
	this->weightsCount = weights != nullptr;
	this->nextCount = 0;
	this->prevCount = 0;

	this->type = LayerBase::LayerType::Basic;
	this->activationType = activationType;
}


BasicLayer::BasicLayer (int length, int width, int height, ActivationType activationType) {
	this->layerMatrixes = new Matrix3D* [1];
   this->layerMatrixes[0] = new Matrix3D (length, width, height);

   this->biasMatrixes = new Matrix3D*[1];
	this->biasMatrixes[0] = nullptr;

   this->weights = (WeightBase**) new BasicWeight*[1];
	this->weights[0] = nullptr;

   this->next = (LayerBase**) new BasicLayer*[1];
	this->next[0] = nullptr;
   this->prev = (LayerBase**) new BasicLayer*[1];
	this->prev[0] = nullptr;

	this->layerMatrixCount = 1;
	this->biasCount = 0;
	this->weightsCount = 0;
	this->nextCount = 0;
	this->prevCount = 0;

	this->type = LayerBase::LayerType::Basic;
	this->activationType = activationType;
}


BasicLayer::BasicLayer () {
	this->layerMatrixes = new Matrix3D* [1];
   this->layerMatrixes[0] = nullptr;
	
   this->biasMatrixes = new Matrix3D*[1];
	this->biasMatrixes[0] = nullptr;

   this->weights = (WeightBase**) new BasicWeight*[1];
	this->weights[0] = nullptr;

   this->next = (LayerBase**) new BasicLayer*[1];
	this->next[0] = nullptr;
   this->prev = (LayerBase**) new BasicLayer*[1];
	this->prev[0] = nullptr;

	this->layerMatrixCount = 0;
	this->biasCount = 0;
	this->weightsCount = 0;
	this->nextCount = 0;
	this->prevCount = 0;

	this->type = LayerBase::LayerType::Basic;
	this->activationType = ActivationType::Sigmoid;
}

BasicLayer::~BasicLayer () { 
   if (this->getLayerMatrix() != nullptr) {
		for (int i = 0; i < this->layerMatrixCount; i++) {
			delete this->getLayerMatrix(i);
		}
   }
   if (this->getBias() != nullptr) {
		for (int i = 0; i < this->biasCount; i++) {
			delete this->getBias(i);
		}
   }
   if (this->getWeights() != nullptr) {
		for (int i = 0; i < this->weightsCount; i++) {
			delete this->getWeights(i);
		}
   }
   if (this->getNext() != nullptr) {
		for (int i = 0; i < this->nextCount; i++) {
			delete this->getNext(i);
		}
	}
}

// broken function
BasicLayer::BasicLayer (const BasicLayer& b, bool copyNext) {
	this->layerMatrixCount = b.getLayerMatrixCount();
	this->biasCount = b.getBiasCount();
	this->weightsCount = b.getWeightsCount();
	this->nextCount = b.getNextCount();
	this->prevCount = b.getPrevCount();

	this->type = b.getLayerType();
	this->activationType = b.getActivationType();

	this->layerMatrixes = new Matrix3D* [1];
   this->layerMatrixes[0] = nullptr;
	if (b.getLayerMatrix() == nullptr) {
		this->layerMatrixes[0] = nullptr;
	} else {
		this->setLayerMatrix(new Matrix3D(*b.getLayerMatrix()));
	}

	this->biasMatrixes = new Matrix3D*[1];
	this->biasMatrixes[0] = nullptr;
	if (b.getBias() != nullptr) {
		this->setBias(new Matrix3D(*(b.getBias())));
	}

	this->weights = (WeightBase**) new BasicWeight*[1];
	this->weights[0] = nullptr;
	if (b.getWeights() != nullptr) {
		this->setWeights(new BasicWeight(*(b.getWeights())));
	}

	this->next = (LayerBase**) new BasicLayer*[1];
	this->next[0] = nullptr;
	this->prev = (LayerBase**) new BasicLayer*[1];
	this->prev[0] = nullptr;

	if (copyNext) {
		const BasicLayer* bCurrent = &b;
		BasicLayer* thisCurrent = this;
		while (bCurrent->getNext() != nullptr) {
			bCurrent = bCurrent->getNext();
			thisCurrent->next = (LayerBase**) new BasicLayer*[1];
			thisCurrent->next[0] = nullptr;
			thisCurrent->setNext(new BasicLayer());
			if (bCurrent->getLayerMatrix() != nullptr) {
				thisCurrent->getNext()->setLayerMatrix(new Matrix3D (*bCurrent->getLayerMatrix()));
			}
			if (bCurrent->getBias() != nullptr) {
				thisCurrent->getNext()->setBias(new Matrix3D (*bCurrent->getBias()));
			}
			if (bCurrent->getWeights() != nullptr) {
				thisCurrent->getNext()->setWeights(new BasicWeight (*bCurrent->getWeights()));
			}
			thisCurrent->getNext()->setPrev(thisCurrent);
			thisCurrent = thisCurrent->getNext();
		}
	}
} 

BasicLayer* BasicLayer::getNext (int index) const {
	return (BasicLayer*) this->LayerBase::getNext(index);
}

BasicWeight* BasicLayer::getWeights (int index) const {
	return (BasicWeight*) this->LayerBase::getWeights(index);
}

// BasicLayer* BasicLayer::add (LayerBase* layer) {
//    if (this->getNext() == nullptr) {
//       this->next[0] = layer;
// 		this->biasMatrixes[0] = this->newBias();
// 		this->weights[0] = this->newWeight();
//    } else {
//       this->next[0] = this->getNext()->add(layer);
//    }
//    return this;
// }

// BasicLayer* BasicLayer::add (Matrix3D* layerMatrix, Matrix3D* biasMatrix, BasicWeight* weights) {
//    if (this->getNext() == nullptr) {
//       this->next[0] = new BasicLayer (layerMatrix, nullptr, nullptr);
//       this->getNext()->setPrev(this);
//       if (this->getBias() == nullptr) {
//          this->biasMatrixes[0] = new Matrix3D(this->getNext()->getLayerMatrix()->getLength(), this->getNext()->getLayerMatrix()->getWidth(), this->getNext()->getLayerMatrix()->getHeight());
//          this->getBias()->randomize(-0.05, 0.05);
//       } else {
//          this->getBias()->setMatrix(biasMatrix);
//       }
		
// 		this->weights[0] = this->newWeight();
//       return this;
//    }
//    this->getNext()->add(layerMatrix, biasMatrix, weights);
//    return this;
// }

WeightBase* BasicLayer::newWeight(int index) {
	if (this->getNext() == nullptr) {
		return nullptr;
	}

	return new BasicWeight (
		this->getLayerMatrix(0)->getLength(), 
		this->getLayerMatrix(0)->getWidth(),
		this->getLayerMatrix(0)->getHeight(),
		this->getNext(index)->getLayerMatrix(0)->getLength(),
		this->getNext(index)->getLayerMatrix(0)->getWidth(),
		this->getNext(index)->getLayerMatrix(0)->getHeight(),
		1);
}

Matrix3D* BasicLayer::newBias(int index) {
	if (this->getNext() == nullptr) {
		return nullptr;
	}

	return new Matrix3D (
		this->getNext()->getLayerMatrix(index)->getLength(),
		this->getNext()->getLayerMatrix(index)->getWidth(),
		this->getNext()->getLayerMatrix(index)->getHeight()
	);
}

void artificialIntelligence::classes::BasicLayer::calculateAndUpdateAllCPU () {
   if (this->getNext(0) == nullptr) {
      return;
   }
   this->calculateAndUpdateLayerCPU();
   this->getNext(0)->calculateAndUpdateAllCPU();
}

void BasicLayer::calculateAndUpdateLayerCPU () {
   Matrix3D* nextLayer = this->getNext(0)->getLayerMatrix(0);
   Matrix3D* outputs = new Matrix3D (nextLayer->getLength(), nextLayer->getWidth(), nextLayer->getHeight());
	outputs->setAll(0);
   if (isnan(*outputs->getData(0, 0, 0))) {
      std::cout << "null init";
      exit (0);
   }
	// nextLayer->printMatrix();

   float activation = 0;
   for (int fl = 0; fl < this->getLayerMatrix(0)->getLength(); fl++) {
      for (int fw = 0; fw < this->getLayerMatrix(0)->getWidth(); fw++) {
         for (int fh = 0; fh < this->getLayerMatrix(0)->getHeight(); fh++) {
            for (int sl = 0; sl < nextLayer->getLength(); sl++) {
               for (int sw = 0; sw < nextLayer->getWidth(); sw++) {
                  for (int sh = 0; sh < nextLayer->getHeight(); sh++) {
                     activation = *this->getLayerMatrix(0)->getData(fl, fw, fh) * *this->getWeights(0)->getData(fl, fw, fh, sl, sw, sh) + *outputs->getData(sl, sw, sh);
                     outputs->insert(activation, sl, sw, sh);
                  }
               }
            }
         }
      }
   } 

	// std::cout << "acc: " << this->activationType << '\n';
	// this->getNext(0)->setLayerMatrix(*this->getNext(0)->getLayerMatrix() + this->getBias(0));
	// activate (this->activationType, this->getNext(0)->getLayerMatrix(0));
	*outputs += this->getBias(0);
	activate(this->activationType, outputs);
   // for (int sl = 0; sl < nextLayer->getLength(); sl++) {
   //    for (int sw = 0; sw < nextLayer->getWidth(); sw++) {
   //       for (int sh = 0; sh < nextLayer->getHeight(); sh++) {
   //          activation = activate(this->activationType, *outputs->getData(sl, sw, sh) + *this->getBias(0)->getData(sl, sw, sh));
   //          outputs->insert(activation, sl, sw, sh);
   //       }
   //    }
   // }

   this->getNext(0)->setLayerMatrix (outputs);
}

void BasicLayer::calculateAndUpdateAllGPUV2() {
	BasicLayer* currentLayer = this;
	Matrix3D* currentLayerMatrix = currentLayer->getLayerMatrix();

	long long numInputs = currentLayerMatrix->getSize() / sizeof(float);
	long long numOutputs = currentLayer->getNext()->getLayerMatrix()->getSize() / sizeof(float);
	long long numWeights = numInputs * numOutputs;
	long long numOutputsRemaining = numOutputs;
	long long outputIndex = 0;

	long long numBlocks = numOutputs > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : numOutputs; 
	long long numThreads = 512;
	long long maxWeightIndex = currentLayer->getWeights()->getWeightMatrix()->getSize() / sizeof(float);
	long long numPerThread = std::ceil ((double)maxWeightIndex / (numBlocks * numThreads));
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

	float* current_weights;
	float* next_weights;

	gpuErrchk(cudaMalloc((void **) &current_weights, maxWeightIndex * sizeof(float)));
	gpuErrchk(cudaMalloc((void **) &next_weights, maxWeightIndex * sizeof(float)));
	gpuErrchk(cudaMemcpy(current_weights, currentWeight->getWeightMatrix(0)->getArr(), maxWeightIndex * sizeof(float), cudaMemcpyHostToDevice));
	weightsInCurrentKernelRun = maxWeightIndex;
	weightsAddedLastSet = maxWeightIndex;
	currentWeightMatrixIndex++;
	
	int startingOutputID = 0;
	int nextOutputID = maxWeightIndex % currentWeight->getOutputSize();
	int numWeightsMatrixesLeft = std::ceil((float)numWeights / maxWeightIndex) - 1;

	gpuErrchk(cudaMemcpy(current_input, input, currentLayerMatrix->getSize(), cudaMemcpyHostToDevice)); 
	gpuErrchk(cudaMemset(current_output, 0b00000000, numOutputs * sizeof(float)));

	int debugCounter = 0;

	numWeightsMatrixesLeft = std::ceil((float)numWeights / maxWeightIndex) - 1;
   while (currentLayer->getNext() != nullptr) {
		currentWeightMatrixIndex = 1;
		outputIndex = 0;
		startingOutputID = 0;
		numOutputsRemaining = numOutputs;
		nextOutputID = weightsAddedLastSet;
		
		bool weightsFinished = false;
		long long weightsUsed = 0;
		do {
			
			if (numWeightsMatrixesLeft >= 1){
				if (currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float) < maxWeightIndex) {
					maxWeightIndex = currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float);
				}
				gpuErrchk(cudaMemcpyAsync(next_weights, currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr(), maxWeightIndex * sizeof(float), cudaMemcpyHostToDevice));
				weightsAddedLastSet = maxWeightIndex;
				currentWeightMatrixIndex++;
				numWeightsMatrixesLeft -= 1;
			} 
			
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
					currentWeightMatrixIndex = 1;
					numWeightsMatrixesLeft = std::ceil((float)nextNumWeights / nextMaxWeightIndex) - 1;
					weightsAddedLastSet = nextMaxWeightIndex;
				}
				weightsFinished = true;
			}
			
			long long helper = 0;

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

					artificialIntelligence::classes::calculateAndUpdateLayerGPUBasic<<< numBlocks, numThreads, sharedSize, stream1 >>>(current_input, current_weights, current_output, numBlocks, numOutputs, numPerThread, weightsInCurrentKernelRun, helper, weightsUsed, startingOutputID);
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
		// printArr(currentWeight->getWeightMatrix(currentWeightMatrixIndex - 1)->getArr(), 10);
		Matrix3D* bias = currentLayer->getBias();
		currentLayer = currentLayer->getNext();
		currentLayerMatrix = currentLayer->getLayerMatrix();
		currentWeight = currentLayer->getWeights();
		numInputs = currentLayerMatrix->getSize() / sizeof(float);

		if (currentLayer->getNext() != nullptr) {
			output = currentLayer->getNext()->getLayerMatrix()->getArr();
			numOutputs = currentLayer->getNext()->getLayerMatrix()->getSize() / sizeof(float);
			numWeights = numInputs * numOutputs;
			maxWeightIndex = currentLayer->getWeights()->getWeightMatrix()->getSize();
			numBlocks = numOutputs > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : numOutputs;
			numThreads = 512; // arbitrary
			numPerThread = std::ceil ((double)maxWeightIndex / (numBlocks * numThreads));
			output = currentLayer->getNext()->getLayerMatrix()->getArr();
			gpuErrchk(cudaFree(next_weights));
			gpuErrchk(cudaMalloc((void **) &next_weights, maxWeightIndex * sizeof(float)));
			gpuErrchk(cudaFree(current_output));
			gpuErrchk(cudaMalloc((void **) &current_output, numOutputs * sizeof(float)));
			gpuErrchk(cudaMemset(current_output, 0b00000000, numOutputs * sizeof(float))); 
		}
		
		*currentLayer->getLayerMatrix() += bias;
		// std::cout << "acc: " << currentLayer->getPrev(0)->getActivationType() << '\n';
		activate (currentLayer->getPrev(0)->getActivationType(), currentLayer->getLayerMatrix(0));
		gpuErrchk(cudaFree(current_input));
		gpuErrchk(cudaMalloc((void **) &current_input, currentLayerMatrix->getSize()));
		input = currentLayerMatrix->getArr();
		gpuErrchk(cudaMemcpy(current_input, input, currentLayerMatrix->getSize(), cudaMemcpyHostToDevice));

		debugCounter++;
	}
	gpuErrchk(cudaFree(current_input));	
	gpuErrchk(cudaFree(current_output));
	gpuErrchk(cudaFree(current_weights));	
	gpuErrchk(cudaFree(next_weights));
	gpuErrchk(cudaStreamDestroy(stream1));
	gpuErrchk(cudaStreamDestroy(stream2));
}

__global__ void artificialIntelligence::classes::calculateAndUpdateLayerGPUBasic(float* nodeValues, float* weights, float* output, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned long long outputNodeId = (blockIdx.x + startingOutputId) % outputSize;
	unsigned int numThreads = blockDim.x;
	unsigned long long weightIndex = tid * outputSize + blockIdx.x + helperIndex;
	unsigned long long inputNodeId = 0;
	unsigned int gridSize = numThreads*outputSize;
	sdata[tid] = 0;

	while (weightIndex < maxWeightIndex) {
		inputNodeId = (weightIndex + startingWeight) / outputSize;
		sdata[tid] += nodeValues[inputNodeId] * weights[weightIndex];
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
		output[outputNodeId] += sdata[0];
	}
}

Matrix3D* BasicLayer::calculateErrorCPU (Matrix3D* delta) {
	Matrix3D* currentLayerMatrix = this->getLayerMatrix();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	for (int l = 0; l < currentLayerMatrix->getLength(); l++) {
		for (int w = 0; w < currentLayerMatrix->getWidth(); w++) {
			for (int h = 0; h < currentLayerMatrix->getHeight(); h++) {
				Matrix3D* outputMatrix = this->getNext(0)->getLayerMatrix(0);
				Matrix3D* weightedMatrix = new Matrix3D (delta->getLength(), delta->getWidth(), delta->getHeight());
				for (int l2 = 0; l2 < outputMatrix->getLength(); l2++) {
					for (int w2 = 0; w2 < outputMatrix->getWidth(); w2++) {
						for (int h2 = 0; h2 < outputMatrix->getHeight(); h2++) {
							weightedMatrix->insert(*this->getWeights(0)->getData(l, w, h, l2, w2, h2) * *delta->getData(l2, w2, h2), l2, w2, h2);
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

	long long numInputs = currentLayerMatrix->getSize() / sizeof(float);
	long long numOutputs = currentLayer->getNext()->getLayerMatrix()->getSize() / sizeof(float);
	long long numWeights = numInputs * numOutputs;
	long long numInputsRemaining = numInputs;
	long long inputIndex = 0;
	long long numBlocks = numInputs > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : numInputs; 
	long long numThreads = 512;
	long long maxWeightIndex = numBlocks * numOutputs;
	long long numPerThread = std::ceil ((double)maxWeightIndex / (numBlocks * numThreads));
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

	cudaStream_t stream1, stream2;
	cudaStreamCreate ( &stream1); 
	cudaStreamCreate ( &stream2); 
	
	BasicWeight* currentWeight = currentLayer->getWeights();
	long long matrixSize = currentWeight->getWeightMatrix()->getSize() / sizeof(float);
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

	int weightsInCurrentMatrix = currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float);
	int weightsInBasicWeight = currentWeight->getSize();

	int numberOfWeightsToAdd = numBlocks * numOutputs;
	int toAdd = weightsInCurrentMatrix > numberOfWeightsToAdd ? numberOfWeightsToAdd : weightsInCurrentMatrix;

	int amountAdded = 0;
	int weightsAdded = 0;

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
		}
	}
	weightsInCurrentKernelRun = weightsAdded;
	weightsAddedLastSet = weightsAdded;
	
	int startingInputID = 0;
	int numWeightsMatrixesLeft = std::ceil((float)numWeights / maxWeightIndex) - 1;

	numWeightsMatrixesLeft = std::ceil((float)numWeights / maxWeightIndex) - 1;
	inputIndex = 0;
	startingInputID = 0;
	numInputsRemaining = numInputs;
	bool weightsFinished = false;
	long long weightsUsed = 0;
	do {
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
			
			artificialIntelligence::classes::calculateErrorBasic<<< numBlocks, numThreads, sharedSize, stream1 >>>(current_weights, current_delta, current_error, numInputs, numOutputs, numPerThread, weightsInCurrentKernelRun, numWeights, weightsUsed, startingInputID);
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
			amountAdded = weightsAdded % (WEIGHT_MAX_SIZE);
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
	gpuErrchk(cudaStreamDestroy(stream1));
	gpuErrchk(cudaStreamDestroy(stream2));
	return errorMatrix;
}

__global__ void artificialIntelligence::classes::calculateErrorBasic(float* weights, float* delta, float* error, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingInputID) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int numThreads = blockDim.x;
	unsigned long long inputNodeId = blockIdx.x + startingInputID;
	unsigned long long weightIndex = tid + blockIdx.x * outputSize;
	unsigned int gridSize = numThreads;
	int weightsToAddStart = outputSize * (blockIdx.x);
	int weightsToAddEnd = outputSize * (blockIdx.x + 1);

	sdata[tid] = 0;
	while (weightIndex >= weightsToAddStart && weightIndex < weightsToAddEnd) {
		sdata[tid] += weights[weightIndex] * delta[(startingWeight + weightIndex) % outputSize];
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
		error[inputNodeId] += sdata[0];
	}
}

void BasicLayer::updateWeightsCPU (Matrix3D* delta, double learningRate) {
	Matrix3D* currentLayerMatrix = this->getLayerMatrix();
	for (int l = 0; l < currentLayerMatrix->getLength(); l++) {
		for (int w = 0; w < currentLayerMatrix->getWidth(); w++) {
			for (int h = 0; h < currentLayerMatrix->getHeight(); h++) {
				float inputValue = *currentLayerMatrix->getData(l, w, h);
				float value = 0;
				
				Matrix3D* weightMatrix = this->getNext()->getLayerMatrix();
				for (int l2 = 0; l2 < weightMatrix->getLength(); l2++) {
					for (int w2 = 0; w2 < weightMatrix->getWidth(); w2++) {
						for (int h2 = 0; h2 < weightMatrix->getHeight(); h2++) {
							value = *this->getWeights()->getData(l, w, h, l2, w2, h2) + inputValue * *delta->getData(l2, w2, h2) * learningRate;
							this->getWeights()->insertData(value, l, w, h, l2, w2, h2);
						}
					}
				}
			}
		}
	}
}

void BasicLayer::updateWeightsGPU (Matrix3D* delta, double learningRate) {
	BasicLayer* currentLayer = this;
	Matrix3D* currentLayerMatrix = currentLayer->getLayerMatrix();

	long long numInputs = currentLayerMatrix->getSize() / sizeof(float);
	long long numOutputs = currentLayer->getNext()->getLayerMatrix()->getSize() / sizeof(float);
	long long numWeights = numInputs * numOutputs;
	long long inputIndex = 0;
	long long numBlocks = numOutputs > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : numOutputs; 
	long long numThreads = 512;
	long long maxWeightIndex = numBlocks * numOutputs;
	long long numPerThread = std::ceil ((double)maxWeightIndex / (numBlocks * numThreads));
	long long sharedSize = numThreads * sizeof(float); 
	if (maxWeightIndex > numWeights) {
		maxWeightIndex = numWeights;
	}
	
	cudaStream_t stream1, stream2;
	cudaStreamCreate ( &stream1); 
	cudaStreamCreate ( &stream2);

	BasicWeight* currentWeight = currentLayer->getWeights();
	long long matrixSize = currentWeight->getWeightMatrix()->getSize() / sizeof(float);
	long long currentWeightMatrixIndex = 0;
	long long weightsInCurrentKernelRun = 0;
	
	int weightsInCurrentMatrix = currentWeight->getWeightMatrix(0)->getSize() / sizeof(float);

	Matrix3D* inputMatrix = currentLayer->getLayerMatrix();
	float* current_input;
	float* current_delta;
	gpuErrchk(cudaMalloc((void **) &current_input, inputMatrix->getSize()));
	gpuErrchk(cudaMalloc((void **) &current_delta, delta->getSize()));
	gpuErrchk(cudaMemcpy(current_input, inputMatrix->getArr(), inputMatrix->getSize(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(current_delta, delta->getArr(), delta->getSize(), cudaMemcpyHostToDevice));
	

	float* current_weights;
	float* next_weights;
	gpuErrchk(cudaMalloc((void **) &current_weights, currentWeight->getWeightMatrix(0)->getSize()));
	gpuErrchk(cudaMalloc((void **) &next_weights, currentWeight->getWeightMatrix(0)->getSize()));
	gpuErrchk(cudaMemcpy(current_weights, currentWeight->getWeightMatrix(0)->getArr(), currentWeight->getWeightMatrix(0)->getSize(), cudaMemcpyHostToDevice));
	weightsInCurrentKernelRun = currentWeight->getWeightMatrix(0)->getSize() / sizeof(float);

	long long weightsUsed = 0;
	int startingInputId = 0;
	while ((numWeights - weightsUsed) != 0) {
		// std::cout << "inside22\n";
		// std::cout << "numBlocks: " << numBlocks << '\n';
		// std::cout << "numOutputs: " << numOutputs << '\n';
		// std::cout << "numPerThread: " << numPerThread << '\n';
		// std::cout << "weightsInCurrentKernelRun: " << weightsInCurrentKernelRun << "\n";
		// std::cout << "numOutputsRemaining: " << numOutputsRemaining << '\n';
		// std::cout << "weightsUsed: " << weightsUsed << "\n";
		// std::cout << "startingInputId: " << startingInputId << "\n\n";

		artificialIntelligence::classes::updateWeightsBasic<<<numBlocks, numThreads, sharedSize, stream1>>>(current_weights, current_delta, current_input, numInputs, numOutputs, numPerThread, weightsInCurrentKernelRun, numWeights, weightsUsed, startingInputId, learningRate);
		inputIndex += numBlocks;
		
		startingInputId = weightsUsed / numOutputs;

		weightsUsed += currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float);

		currentWeightMatrixIndex++;
		if ((numWeights - weightsUsed) != 0) {
			gpuErrchk(cudaMemcpyAsync(next_weights, currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr(), currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize(), cudaMemcpyHostToDevice));
			weightsInCurrentKernelRun = currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float);
		}

		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(currentWeight->getWeightMatrix(currentWeightMatrixIndex - 1)->getArr(), current_weights, currentWeight->getWeightMatrix(currentWeightMatrixIndex - 1)->getSize(), cudaMemcpyDeviceToHost));

		float* temp = current_weights;
		current_weights = next_weights;
		next_weights = temp;
	}

	gpuErrchk(cudaFree(current_input));
	gpuErrchk(cudaFree(current_delta));
	gpuErrchk(cudaFree(current_weights));	
	gpuErrchk(cudaFree(next_weights));
	gpuErrchk(cudaStreamDestroy(stream1));
	gpuErrchk(cudaStreamDestroy(stream2));
}

__global__ void artificialIntelligence::classes::updateWeightsBasic(float* weights, float* delta, float* input, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingInputID, double learningRate) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int numThreads = blockDim.x;
	unsigned long long weightIndex = tid + numThreads * blockIdx.x;
	unsigned long long outputNodeId = (weightIndex + startingWeight) % outputSize;
	unsigned long long inputNodeId = (weightIndex + startingWeight) / outputSize;
	unsigned int gridSize = numThreads * gridDim.x;
	while (weightIndex < maxWeightIndex) {
		weights[weightIndex] += input[inputNodeId] * delta[outputNodeId] * learningRate;
		weightIndex += gridSize;
		inputNodeId = (weightIndex + startingWeight) / outputSize;
		outputNodeId = (weightIndex + startingWeight) % outputSize;
	}
}

void BasicLayer::printDetails () {
	std::cout << "Basic Fully Connected Layer :: ";
	this->getLayerMatrix()->printMatrixSize();
}  


void BasicLayer::toFile (std::ofstream* outputFile) {
	char* output = new char[sizeof(int) * 6];

	//layer
	*outputFile << this->type << '\n';
   *outputFile << this->getLayerMatrix()->getLength() << ',' << this->getLayerMatrix()->getWidth() << ',' << this->getLayerMatrix()->getHeight() << '\n';

	//bias
   if (this->biasMatrixes[0] == nullptr) {
      return;
   }
   *outputFile << this->getBias()->getLength() << ',' << this->getBias()->getWidth() << ',' << this->getBias()->getHeight() << '\n';
   for (int i = 0; i < this->getBias()->getLength(); i++) {
      for (int j = 0; j < this->getBias()->getWidth(); j++) {
         for (int k = 0; k < this->getBias()->getHeight(); k++) {
            *outputFile << *this->getBias()->getData(i, j, k) << ',';
         }
      }
   }

   outputFile->seekp((int) outputFile->tellp() - 1);
   outputFile->write("\n", 1);

	//weights
   if (this->weights[0] == nullptr) {
      return;
   }
   *outputFile << this->getLayerMatrix()->getLength() << ',' << this->getLayerMatrix()->getWidth() << ',' << this->getLayerMatrix()->getHeight() << ',';
   *outputFile << this->getBias()->getLength() << ',' << this->getBias()->getWidth() << ',' << this->getBias()->getHeight() << '\n';

	int currentWeightMatrix = 0;
	float* weights;
	while (this->getWeights()->getWeightMatrix(currentWeightMatrix) != nullptr) {
		int size = this->getWeights()->getWeightMatrix(currentWeightMatrix)->getSize() + sizeof(float);
		char* output = new char[size];
		char* ptr = output;

		weights = this->getWeights()->getWeightMatrix(currentWeightMatrix)->getArr();
		for (int i = 0, cc = this->getWeights()->getWeightMatrix(currentWeightMatrix)->getSize() / sizeof(float); i < cc; i++) {
			memcpy(ptr, &weights[i], sizeof(float));
			ptr += sizeof(float);
		}
		outputFile->write(output, size);
		currentWeightMatrix++;

		free(output);
	}
	outputFile->seekp((int) outputFile->tellp() - 1);
   outputFile->write("\n", 1);

	//activation
	*outputFile << (int) this->activationType << '\n';

   if (this->getNext() == nullptr) {
      return;
   }
   this->getNext()->toFile(outputFile);
}


LayerBase* BasicLayer::loadFromFile (std::ifstream* inputFile, LayerBase* prev) {
	std::cout << "Loading layer from file\n";
   BasicLayer* layer = new BasicLayer ();
   std::string line;
   getline (*inputFile, line);
	std::cout << "line: " << line << '\n';
   std::stringstream lineStream;
   lineStream << line;
   std::string value;
   getline(lineStream, value, ',');
	std::cout << "v1: " << value << '\n';
   int layerLength = stoi(value);
   getline(lineStream, value, ',');
	std::cout << "v2: " << value << '\n';
   int layerWidth = stoi(value);
   getline(lineStream, value, ',');
	std::cout << "v3: " << value << '\n';
   int layerHeight = stoi(value);
   layer->setLayerMatrix(new Matrix3D (layerLength, layerWidth, layerHeight), 0);
   layer->setPrev(prev, 0);

   lineStream.str(std::string());
   lineStream.clear();
   getline (*inputFile, line);
   lineStream << line;

   if (inputFile->eof()) {
		layer->biasMatrixes = new Matrix3D*[1];
   	layer->weights = (WeightBase**) new BasicWeight*[1];
      return layer;
   }

   getline(lineStream, value, ',');
   int biasLength = stoi(value);
   getline(lineStream, value, ',');
   int biasWidth = stoi(value);
   getline(lineStream, value, ',');
   int biasHeight = stoi(value);
   layer->setBias (new Matrix3D (biasLength, biasWidth, biasHeight));

   lineStream.str(std::string());
   lineStream.clear();
   getline (*inputFile, line);
   lineStream << line;
   for (int i = 0; i < layer->getBias(0)->getLength(); i++) {
      for (int j = 0; j < layer->getBias(0)->getWidth(); j++) {
         for (int k = 0; k < layer->getBias(0)->getHeight(); k++) {
            std::getline(lineStream, value, ',');
            layer->getBias(0)->insert (stod(value), i, j, k);
         }
      }
   }

   getline (*inputFile, line);

   if (inputFile->eof()) {
		layer->weights = (WeightBase**) new BasicWeight*[1];
      return layer;
   }

   BasicWeight* weights = new BasicWeight (
      layer->getLayerMatrix()->getLength(), 
      layer->getLayerMatrix()->getWidth(), 
      layer->getLayerMatrix()->getHeight(), 
      layer->getBias()->getLength(), 
      layer->getBias()->getWidth(), 
      layer->getBias()->getHeight(),
		0
   );

   
   lineStream.str(std::string());
   lineStream.clear();
	
	std::cout << "Inserting weights\n";

	int currentWeightMatrix = 0;
	while (weights->getWeightMatrix(currentWeightMatrix) != nullptr) {
		inputFile->read((char*) weights->getWeightMatrix(currentWeightMatrix)->getArr(), weights->getWeightMatrix(currentWeightMatrix)->getSize());
		currentWeightMatrix++;
	}
	getline(*inputFile, line);

	std::cout << "Finished weights\n";

   layer->setWeights (weights);
	std::cout << line << '\n';
	std::cout << "here\n";
   getline (*inputFile, line);
	std::cout << line << '\n';
	std::cout << "here\n";
   layer->setActivation((ActivationType) stoi(line));

   return layer;
}