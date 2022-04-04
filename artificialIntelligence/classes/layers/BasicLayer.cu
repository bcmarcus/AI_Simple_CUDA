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

#include "../../functions/activationFunctions.cuh"
#include "../layers/BasicLayer.cuh"
#include "../weights/BasicWeight.cuh"

using namespace std;
using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::debug;
using namespace artificialIntelligence::classes;
using namespace artificialIntelligence::functions::activation;

#define MAX_BLOCK_SIZE 8192

BasicLayer::BasicLayer (Matrix3D* layerMatrix, Matrix3D* biasMatrix, BasicWeight* weights) {
   this->layerMatrix = new Matrix3D(layerMatrix->getLength(), layerMatrix->getWidth(), layerMatrix->getHeight());
   this->getLayer()->setMatrix(layerMatrix);
	this->biasMatrixes = new Matrix3D*[1];
	this->biasMatrixes[0] = nullptr;
   if (biasMatrix != nullptr) {
		std::cout << "here\n";
      this->biasMatrixes[0] = new Matrix3D(biasMatrix->getLength(), biasMatrix->getWidth(), biasMatrix->getHeight());
      this->getBias()->setMatrix(biasMatrix);
   }
	this->weights = (WeightBase**) new BasicWeight*[1];
   this->weights[0] = weights;
   this->next = (LayerBase**) new BasicLayer*[1];
	this->next[0] = nullptr;
   this->prev = (LayerBase**) new BasicLayer*[1];
	this->prev[0] = nullptr;
}


BasicLayer::BasicLayer (int length, int width, int height) {
   this->layerMatrix = new Matrix3D (length, width, height);
   this->getLayer()->randomize();
   this->biasMatrixes = new Matrix3D*[1];
	this->biasMatrixes[0] = nullptr;
   this->weights = (WeightBase**) new BasicWeight*[1];
   this->next = (LayerBase**) new BasicLayer*[1];
	this->next[0] = nullptr;
   this->prev = (LayerBase**) new BasicLayer*[1];
	this->prev[0] = nullptr;
}


BasicLayer::BasicLayer () {
   this->layerMatrix = nullptr;
   this->biasMatrixes = new Matrix3D*[1];
	this->biasMatrixes[0] = nullptr;
   this->weights = (WeightBase**) new BasicWeight*[1];
	this->weights[0] = nullptr;
   this->next = (LayerBase**) new BasicLayer*[1];
	this->next[0] = nullptr;
   this->prev = (LayerBase**) new BasicLayer*[1];
	this->prev[0] = nullptr;
}


BasicLayer::~BasicLayer () { 
   if (this->layerMatrix != nullptr) {
      delete this->getLayer();
   }
   if (this->getBias() != nullptr) {
      delete this->getBias();
   }
   if (this->getWeights() != nullptr) {
      delete this->getWeights();
   }
   if (this->getNext() != nullptr) {
		delete this->getNext();
	}
}

// broken function
BasicLayer::BasicLayer (const BasicLayer& b, bool copyNext) {
	if (b.getLayer() == nullptr) {
		this->layerMatrix = nullptr;
	} else {
		this->layerMatrix = new Matrix3D(*b.getLayer());
	}
	this->biasMatrixes = new Matrix3D*[1];
	this->biasMatrixes[0] = nullptr;
	if (b.getBias() != nullptr) {
		this->biasMatrixes[0] = new Matrix3D(*(b.getBias()));
	}
	this->weights = (WeightBase**) new BasicWeight*[1];
	this->weights[0] = nullptr;
	if (b.getWeights() != nullptr) {
		this->weights[0] = new BasicWeight(*(b.getWeights()));
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
			thisCurrent->next[0] = new BasicLayer();
			if (bCurrent->getLayer() != nullptr) {
				thisCurrent->getNext()->setLayer(new Matrix3D (*bCurrent->getLayer()));
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

BasicLayer* BasicLayer::getNext () const {
   return ((BasicLayer*) this->LayerBase::getNext(0));
}

BasicLayer* BasicLayer::getPrev () const {
   return ((BasicLayer*) this->LayerBase::getPrev(0));
}

BasicLayer* BasicLayer::getLast () {
   if (this->LayerBase::getNext(0) == nullptr) {
      return this;
   }
   return ((BasicLayer*) this->LayerBase::getNext(0))->getLast();
}

Matrix3D* BasicLayer::getLayer () const {
   return this->LayerBase::getLayer();
}

Matrix3D* BasicLayer::getBias () const {
   return this->LayerBase::getBias(0);
}

BasicWeight* BasicLayer::getWeights () const {
	return (BasicWeight*) this->LayerBase::getWeights(0);
}

void BasicLayer::setNext (BasicLayer* next) {
   if (this->getNext() != nullptr) {
      delete this->getNext();
   }
   this->next[0] = next;
} 

void BasicLayer::setPrev (BasicLayer* prev) {
   if (this->getPrev() != nullptr) {
      delete this->getPrev();
   }
   this->prev[0] = prev;
} 

void BasicLayer::setLayer (Matrix3D* layerMatrix) {
   if (this->getLayer() == nullptr) {
      this->layerMatrix = new Matrix3D (layerMatrix->getLength(), layerMatrix->getWidth(), layerMatrix->getHeight());
   }
   this->getLayer()->setMatrix(layerMatrix);
}

void BasicLayer::setBias (Matrix3D* biasMatrix) {
   if (this->getBias() == nullptr) {
      this->biasMatrixes[0] = new Matrix3D (biasMatrix->getLength(), biasMatrix->getWidth(), biasMatrix->getHeight());
   }
   this->getBias()->setMatrix(biasMatrix);
}

void BasicLayer::setWeights (BasicWeight* weights) {
   if (this->getWeights() != nullptr) {
      delete this->getWeights();
   }
   this->weights[0] = weights;
}

BasicLayer* BasicLayer::add (BasicLayer* layer) {
   if (this->getNext() == nullptr) {
      this->next[0] = layer;
   } else {
      this->next[0] = this->getNext()->add(layer);
   }
   return this;
}

BasicLayer* BasicLayer::add (Matrix3D* layerMatrix, Matrix3D* biasMatrix, BasicWeight* weights) {
   if (this->getNext() == nullptr) {
      this->next[0] = new BasicLayer (layerMatrix, nullptr, nullptr);
      this->getNext()->setPrev(this);
      if (this->getBias() == nullptr) {
         this->biasMatrixes[0] = new Matrix3D(this->getNext()->getLayer()->getLength(), this->getNext()->getLayer()->getWidth(), this->getNext()->getLayer()->getHeight());
         this->getBias()->randomize(-0.05, 0.05);
      } else {
         this->getBias()->setMatrix(biasMatrix);
      }
		
		this->weights[0] = this->newWeight(this, this->getNext());
      return this;
   }
   this->getNext()->add(layerMatrix, biasMatrix, weights);
   return this;
}

BasicWeight* BasicLayer::newWeight(BasicLayer* firstLayer, BasicLayer* secondLayer) {
	return new BasicWeight (
		firstLayer->getLayer()->getLength(), 
		firstLayer->getLayer()->getWidth(),
		firstLayer->getLayer()->getHeight(),
		secondLayer->getLayer()->getLength(),
		secondLayer->getLayer()->getWidth(),
		secondLayer->getLayer()->getHeight(),
		1);
}

void artificialIntelligence::classes::BasicLayer::calculateAndUpdateAllCPU () {
   if (this->getNext() == nullptr) {
      return;
   }
   this->calculateAndUpdateLayerCPU();
   this->getNext()->calculateAndUpdateAllCPU();
}

void BasicLayer::calculateAndUpdateLayerCPU () {
   Matrix3D* nextLayer = this->getNext()->getLayer();
   Matrix3D* outputs = new Matrix3D (nextLayer->getLength(), nextLayer->getWidth(), nextLayer->getHeight());
	outputs->setAll(0);
   if (isnan(*outputs->getData(0, 0, 0))) {
      std::cout << "null init";
      exit (0);
   }
   float activation = 0;
   for (int fl = 0; fl < this->getLayer()->getLength(); fl++) {
      for (int fw = 0; fw < this->getLayer()->getWidth(); fw++) {
         for (int fh = 0; fh < this->getLayer()->getHeight(); fh++) {
            for (int sl = 0; sl < nextLayer->getLength(); sl++) {
               for (int sw = 0; sw < nextLayer->getWidth(); sw++) {
                  for (int sh = 0; sh < nextLayer->getHeight(); sh++) {
                     activation = *this->getLayer()->getData(fl, fw, fh) * *this->getWeights()->getData(fl, fw, fh, sl, sw, sh) + *outputs->getData(sl, sw, sh);
                     outputs->insert(activation, sl, sw, sh);
                  }
               }
            }
         }
      }
   } 

   for (int sl = 0; sl < nextLayer->getLength(); sl++) {
      for (int sw = 0; sw < nextLayer->getWidth(); sw++) {
         for (int sh = 0; sh < nextLayer->getHeight(); sh++) {
            activation = sigmoid(*outputs->getData(sl, sw, sh) + *this->getBias()->getData(sl, sw, sh));
            outputs->insert(activation, sl, sw, sh);
         }
      }
   }

   this->getNext()->setLayer (outputs);
   delete outputs;
}

void BasicLayer::calculateAndUpdateAllGPUV2() {
	BasicLayer* currentLayer = this;
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();

	long long numInputs = currentLayerMatrix->getSize() / sizeof(float);
	long long numOutputs = currentLayer->getNext()->getLayer()->getSize() / sizeof(float);
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
	float* output = currentLayer->getNext()->getLayer()->getArr();
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
					int nextNumWeights = numOutputs * currentLayer->getNext()->getNext()->getLayer()->getSize() / sizeof(float);
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
		// printArr(currentWeight->getWeightMatrix(currentWeightMatrixIndex - 1)->getArr(), 10);
		Matrix3D* bias = currentLayer->getBias();
		currentLayer = currentLayer->getNext();
		currentLayerMatrix = currentLayer->getLayer();
		currentWeight = currentLayer->getWeights();
		numInputs = currentLayerMatrix->getSize() / sizeof(float);

		if (currentLayer->getNext() != nullptr) {
			output = currentLayer->getNext()->getLayer()->getArr();
			numOutputs = currentLayer->getNext()->getLayer()->getSize() / sizeof(float);
			numWeights = numInputs * numOutputs;
			maxWeightIndex = currentLayer->getWeights()->getWeightMatrix()->getSize();
			numBlocks = numOutputs > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : numOutputs;
			numThreads = 512; // arbitrary
			numPerThread = std::ceil ((double)maxWeightIndex / (numBlocks * numThreads));
			output = currentLayer->getNext()->getLayer()->getArr();
			gpuErrchk(cudaFree(next_weights));
			gpuErrchk(cudaMalloc((void **) &next_weights, maxWeightIndex * sizeof(float)));
			gpuErrchk(cudaFree(current_output));
			gpuErrchk(cudaMalloc((void **) &current_output, numOutputs * sizeof(float)));
			gpuErrchk(cudaMemset(current_output, 0b00000000, numOutputs * sizeof(float))); 
		}
		
		*currentLayer->getLayer() += bias;
		sigmoid(currentLayer->getLayer(), false);
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

__global__ void artificialIntelligence::classes::calculateAndUpdateLayerGPU(float* nodeValues, float* weights, float* output, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId) {
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
	Matrix3D* currentLayerMatrix = this->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	for (int l = 0; l < currentLayerMatrix->getLength(); l++) {
		for (int w = 0; w < currentLayerMatrix->getWidth(); w++) {
			for (int h = 0; h < currentLayerMatrix->getHeight(); h++) {
				Matrix3D* outputMatrix = this->getNext()->getLayer();
				Matrix3D* weightedMatrix = new Matrix3D (delta->getLength(), delta->getWidth(), delta->getHeight());
				for (int l2 = 0; l2 < outputMatrix->getLength(); l2++) {
					for (int w2 = 0; w2 < outputMatrix->getWidth(); w2++) {
						for (int h2 = 0; h2 < outputMatrix->getHeight(); h2++) {
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
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();

	long long numInputs = currentLayerMatrix->getSize() / sizeof(float);
	long long numOutputs = currentLayer->getNext()->getLayer()->getSize() / sizeof(float);
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
	Matrix3D* errorMatrix = new Matrix3D(currentLayer->getLayer()->getLength(), currentLayer->getLayer()->getWidth(), currentLayer->getLayer()->getHeight());
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
	Matrix3D* currentLayerMatrix = this->getLayer();
	for (int l = 0; l < currentLayerMatrix->getLength(); l++) {
		for (int w = 0; w < currentLayerMatrix->getWidth(); w++) {
			for (int h = 0; h < currentLayerMatrix->getHeight(); h++) {
				float inputValue = *currentLayerMatrix->getData(l, w, h);
				float value = 0;
				
				Matrix3D* weightMatrix = this->getNext()->getLayer();
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
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();

	long long numInputs = currentLayerMatrix->getSize() / sizeof(float);
	long long numOutputs = currentLayer->getNext()->getLayer()->getSize() / sizeof(float);
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

	Matrix3D* inputMatrix = currentLayer->getLayer();
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

		artificialIntelligence::classes::updateWeights<<<numBlocks, numThreads, sharedSize, stream1>>>(current_weights, current_delta, current_input, numInputs, numOutputs, numPerThread, weightsInCurrentKernelRun, numWeights, weightsUsed, startingInputId, learningRate);
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

__global__ void artificialIntelligence::classes::updateWeights(float* weights, float* delta, float* input, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingInputID, double learningRate) {
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

int BasicLayer::print (bool printBias, bool printWeights, int depth) {
   if (this->layerMatrix != nullptr) {
      std::cout << "\n\nCurrent Index: " << depth << '\n';
      std::cout << "Layer Matrix: \n";
      this->getLayer()->printMatrix();
   } else {
      std::cout << "No layer found!\n";
      return depth;
   }
   if (printBias) {
      if (this->biasMatrixes[0] != nullptr) {
         std::cout << "Bias Matrix: \n";
         this->getBias()->printMatrix();
      } else {
         std::cout << "No biases found!\n";
      }
   }
   if (printWeights) {
      if (this->weights[0] != nullptr) {
			std::cout << "Weight Matrix: \n";
         this->getWeights()->print();
      } else {
         std::cout << "No weights found!\n";
      }
   }
   if (this->getNext() == nullptr) {
      return depth;
   }
   return this->getNext()->print(printBias, printWeights, depth + 1);
}  


void BasicLayer::toFile (std::ofstream* outputFile) {
	char* output = new char[sizeof(int) * 6];
   *outputFile << this->getLayer()->getLength() << ',' << this->getLayer()->getWidth() << ',' << this->getLayer()->getHeight() << '\n';

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

   if (this->weights[0] == nullptr) {
      return;
   }

   *outputFile << this->getLayer()->getLength() << ',' << this->getLayer()->getWidth() << ',' << this->getLayer()->getHeight() << ',';
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

   if (this->getNext() == nullptr) {
      return;
   }
   this->getNext()->toFile(outputFile);
}


BasicLayer* BasicLayer::loadFromFile (std::ifstream* inputFile, BasicLayer* prev) {
	std::cout << "Loading layer from file\n";
   BasicLayer* layer = new BasicLayer ();
   std::string line;
   getline (*inputFile, line);
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
   Matrix3D* layerMatrix = new Matrix3D (layerLength, layerWidth, layerHeight);
   layer->layerMatrix = layerMatrix;
   layer->prev[0] = prev;

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
   Matrix3D* biasMatrix = new Matrix3D (biasLength, biasWidth, biasHeight);
   layer->biasMatrixes[0] = biasMatrix;

   lineStream.str(std::string());
   lineStream.clear();
   getline (*inputFile, line);
   lineStream << line;
   for (int i = 0; i < layer->getBias()->getLength(); i++) {
      for (int j = 0; j < layer->getBias()->getWidth(); j++) {
         for (int k = 0; k < layer->getBias()->getHeight(); k++) {
            std::getline(lineStream, value, ',');
            layer->getBias()->insert (stod(value), i, j, k);
         }
      }
   }

   getline (*inputFile, line);

   if (inputFile->eof()) {
		layer->weights = (WeightBase**) new BasicWeight*[1];
      return layer;
   }

   BasicWeight* weights = new BasicWeight (
      layer->getLayer()->getLength(), 
      layer->getLayer()->getWidth(), 
      layer->getLayer()->getHeight(), 
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

   layer->weights[0] = weights;
   layer->next[0] = BasicLayer::loadFromFile (inputFile, layer);

   return layer;
}