#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <coreutils/util/time.hpp>

#include <coreutils/util/cudaErrors.cuh>
#include <coreutils/functions/debug/print.cpp>

#include <artificialIntelligence/functions/activationFunctions.cuh>
#include <artificialIntelligence/classes/BasicLayer.cuh>
#include <artificialIntelligence/classes/BasicWeight.cuh>

using namespace std;
using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::debug;
using namespace artificialIntelligence::classes;
using namespace artificialIntelligence::functions::activation;


// a weight list of lists of lists of Matrix3D


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
   if (this->prev != nullptr) {
      this->prev->next = this->next;
   }
}

// 
// BasicLayer::~BasicLayer () {
//    if (this->next != nullptr) {
//       delete this->next;
//    }
// }


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

//if it hits the end, it adds a new one to the back of the list and then
//returns the newly added node, along with a way to tell the previous node the weights needed

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

void BasicLayer::calculateAndUpdateAllGPU () {
	BasicLayer* currentLayer = this;
	Matrix3D* currentLayerMatrix = currentLayer->getLayerMatrix();
	// currentLayer->weights->print();

	int numNodes = currentLayerMatrix->getSize() / sizeof(float); // the number of blocks that will be generated
	int numOutputs = currentLayer->next->getLayerMatrix()->getSize() / sizeof(float);
	int numWeights = numNodes * numOutputs;
	int numOutputsRemaining = numOutputs;
	int numWeightsRemaining = numWeights;
	int outputIndex = 0;

	// :::: TESTING SETTINGS :::: //

	// gpu kernel values
	// int numPerThread = 3;
	// int numBlocks = numOutputs / 2;
	// int numThreads = 1;

	// setting up starting values 
	// int maxWeightIndex = 15;

	// :::: REAL SETTINGS :::: //
	int numPerThread = (int) log2 (numWeights) * 4; // making each node do log2(n) * 4 amount of work because each weight is one work and they run fairly fast
	int numBlocks = numOutputs > 8192 ? 8192 : numOutputs; // break the number of blocks into chunks of 16384 or less
	int numThreads = 256; // arbitrary

	int maxWeightIndex = numBlocks * numThreads * numPerThread; // number of weights per iteration

	if (maxWeightIndex > numWeights) {
		maxWeightIndex = numWeights;
	}


	int sharedSize = numThreads * sizeof(float); // sharedSize within the block should be same as number of threads, because each thread only uses one element in the shared dataset
	
	// input nodes and weights
	float* input = currentLayerMatrix->getArr();
	float* output = currentLayer->next->getLayerMatrix()->getArr();

	// two sets so that speed can be increased 
	float* current_input;
	float* current_output;
	float* device_weights1;
	float* device_weights2; 

	// streams for asynchronous
	cudaStream_t stream1, stream2;
	cudaStreamCreate ( &stream1); 
	cudaStreamCreate ( &stream2); 

	// two sets so that speed can be increased
	// only the weights are host because the others will be fast due to small sizes
		

	gpuErrchk(cudaMalloc((void **) &current_input, currentLayerMatrix->getSize()));
	gpuErrchk(cudaMalloc((void **) &device_weights1, maxWeightIndex * sizeof(float)));
	gpuErrchk(cudaMalloc((void **) &current_output, numOutputs * sizeof(float)));
	// gpuErrchk(cudaMalloc((void **) &device_inputs2, currentLayerMatrix->getSize()));
	gpuErrchk(cudaMalloc((void **) &device_weights2, maxWeightIndex * sizeof(float)));
	// gpuErrchk(cudaMalloc((void **) &device_output2, numOutputs * sizeof(float)));

	float* current_weights = device_weights1;
	float* next_weights = device_weights2;


	// do first memcpy for weights
	long long matrixSize = currentLayer->weights->weights->getSize() / sizeof(float);

	BasicWeight* currentWeight = currentLayer->weights;
	int numLeftToAdd = maxWeightIndex;
	int weightIndex = 0;



	std::cout << "Number of threads: " << numThreads << '\n';
	std::cout << "Number of blocks: " << numBlocks << '\n';
	std::cout << "Number per thread: " << numPerThread << '\n';
	std::cout << "Number of bytes for shared storage: " << sharedSize << "\n";
	std::cout << "Max array index: " << maxWeightIndex << "\n";
	std::cout << "Max byte index: " << maxWeightIndex * sizeof(float) << "\n";

	int numLastAdded = 0;
	int currentWeightMatrixIndex = 0;
	int weightsAddedLastSet = 0;
	// std::cout << "numLeftToAdd" <<  numLeftToAdd << "\n";
	if (numLeftToAdd == 0) {
		return;
	}

	while (numLeftToAdd > 0) {
		if (matrixSize <= numLeftToAdd) {
			// gpuErrchk(cudaMemcpy(&current_weights[weightIndex], &currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr()[0], matrixSize * sizeof(float), cudaMemcpyHostToDevice));
			weightIndex += matrixSize;
			numLeftToAdd -= matrixSize;
			currentWeightMatrixIndex++;
		} else {
			// gpuErrchk(cudaMemcpy(&current_weights[weightIndex], &currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr()[0], numLeftToAdd * sizeof(float), cudaMemcpyHostToDevice));
			numLastAdded = numLeftToAdd;
			numLeftToAdd = 0;
			weightIndex += numLeftToAdd;
			break;
		}
	}
	// currentWeight->print();
	// currentLayer->next->next->weights->print();
	// exit(0);
	int startingOutputID = 0;
	int nextOutputID = numLastAdded;
	weightsAddedLastSet = weightIndex;
	std::cout << "\n\n\n";
	// go through every single layer
   while (currentLayer->next != nullptr) {
		outputIndex = 0;
		startingOutputID = 0;
		matrixSize = currentLayer->weights->weights->getSize() / sizeof(float);
		// copy the inputs and outputs to device
		input = currentLayerMatrix->getArr();
		output = currentLayer->next->getLayerMatrix()->getArr();
		gpuErrchk(cudaMemcpy(current_input, input, currentLayerMatrix->getSize(), cudaMemcpyHostToDevice)); // set the input layer to the input
		gpuErrchk(cudaMemset(current_output, 0b00000000, numOutputs * sizeof(float))); // set the output to zero
		bool weightsFinished = false;
		do {
			weightIndex = 0;
			numWeightsRemaining -= maxWeightIndex;
			if (numWeightsRemaining <= 0) {
				if (currentLayer->next->next != nullptr) {
					currentWeight = currentLayer->next->weights;
 					// matrixSize = currentWeight->getWeightMatrix()->getSize() / sizeof(float);
					// numLastAdded = 0;
					currentWeightMatrixIndex = 0;
					numWeightsRemaining = numOutputs * currentLayer->next->next->getLayerMatrix()->getSize() / sizeof(float);
				} else {
					numLastAdded = matrixSize;
					numWeightsRemaining = 0;
				}
				weightsFinished = true;
			}

			// std::cout << "numWeightsRemaining: " << numWeightsRemaining << '\n';
			// std::cout << "numOutputsRemaining: " << numOutputsRemaining << "\n";

			numOutputsRemaining = numOutputs;

			if (numWeightsRemaining < maxWeightIndex) {
				// std::cout << numWeightsRemaining << "here\n";
				numLeftToAdd = numWeightsRemaining;
			} else {
				numLeftToAdd = maxWeightIndex;
			}
			// run as many times as required until no more output nodes exist


			// max weight Index = 3;
			// 3, 3, 2
			// 3, 6, 8
			// this means that it will take only part of a single matrix
			if (matrixSize - numLastAdded > maxWeightIndex) {
				// std::cout << "inside21\n";
				// std::cout << "weightIndex: " << weightIndex << "\n";
				// std::cout << "Max array index: " << maxWeightIndex << "\n";
				// std::cout << "numLeftToAdd: " << numLeftToAdd << '\n';
				// std::cout << "numLastAdded: " << numLastAdded << '\n';
				// std::cout << "matrixSize - numLastAdded: " << matrixSize - numLastAdded << '\n';

				// does not extend past the length of the matrix
				int valToAdd = matrixSize - numLastAdded < numLeftToAdd ? matrixSize - numLastAdded : numLeftToAdd;

				if (valToAdd != 0) { 
					// gpuErrchk(cudaMemcpy(&next_weights[weightIndex], &currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr()[numLastAdded], valToAdd * sizeof(float), cudaMemcpyHostToDevice));

					// the index doesnt even matter
					weightIndex = valToAdd;

					// numLeft is zero
					numLeftToAdd = 0;

					// amount last added is the same as above
					numLastAdded += valToAdd;

					// index is now at the next point
					nextOutputID += valToAdd;
				}
				// std::cout << "nextOutputID: " << nextOutputID << '\n';
			} 

			// extends beyond the length of the matrix
			else {

				// std::cout << "inside22\n";
				// std::cout << "weightIndex: " << weightIndex << "\n";
				// std::cout << "numLeftToAdd: " << numLeftToAdd << '\n';
				// std::cout << "numLastAdded: " << numLastAdded << '\n';
				// std::cout << "matrixSize - numLastAdded: " << matrixSize - numLastAdded << '\n';

				int valToAdd = matrixSize - numLastAdded < numLeftToAdd ? matrixSize - numLastAdded : numLeftToAdd;

				if (valToAdd != 0) { 
					// gpuErrchk(cudaMemcpy(&next_weights[weightIndex], &currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr()[numLastAdded], (valToAdd) * sizeof(float), cudaMemcpyHostToDevice));

					// current index is just however many were added
					weightIndex = valToAdd;

					// total that should be added - however many were added
					numLeftToAdd -= valToAdd;

					// nextOutputID = the starting point of the matrix
					nextOutputID = numLastAdded;

					// this can now safely be set to zero
					numLastAdded = 0;

					// go to the next matrix
					currentWeightMatrixIndex++;
				}
				// std::cout << "nextOutputID: " << nextOutputID << '\n';
			}


			// std::cout << "\nnumOutputsRemaining: " << numOutputsRemaining << "\n";
			// small set of input nodes output = currentLayer->next->getLayerMatrix()->getArr();correlating to all output nodes
			do {
				// if (numWeightsRemaining == 3) {
				// 	std::cout << numLeftToAdd << '\n';
				// 	exit(0);
				// }
				//copy the next set of weights into the cache while running the kernel
				if (numLeftToAdd > 0) {

					// a full matrix was loaded
					if (matrixSize <= numLeftToAdd) {
						// std::cout << "inside11\n";
						// std::cout << "weightIndex: " << weightIndex << "\n";
						// std::cout << "numLeftToAdd: " << numLeftToAdd << '\n';
						// std::cout << "numLastAdded: " << numLastAdded << '\n';
						// std::cout << "matrixSize - numLastAdded: " << matrixSize - numLastAdded << "\n\n";

						// add the full matrix
						// gpuErrchk(cudaMemcpyAsync(&next_weights[weightIndex], &currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr()[0], matrixSize * sizeof(float), cudaMemcpyHostToDevice, stream2));
						// update the weightIndex so its added in the right spot next time
						weightIndex += matrixSize;

						// subtracts the amount left to add
						numLeftToAdd -= matrixSize;

						// moves to the next matrix
						currentWeightMatrixIndex++;
					} 

					// a full matrix was not loaded
					else {
						// std::cout << "inside12\n";
						// std::cout << "weightIndex: " << weightIndex << "\n";
						// std::cout << "numLeftToAdd: " << numLeftToAdd << '\n';
						// std::cout << "numLastAdded: " << numLastAdded << '\n';
						// std::cout << "matrixSize - numLastAdded: " << matrixSize - numLastAdded << "\n\n";

						// copy the first part of the matrix over
						// gpuErrchk(cudaMemcpyAsync(&next_weights[weightIndex], &currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr()[0], numLeftToAdd * sizeof(float), cudaMemcpyHostToDevice, (cudaStream_t)0));
						// this is just numLeftToAdd now
						numLastAdded = numLeftToAdd;

						// used to make the calculate work for next set
						weightIndex += numLeftToAdd;
						
						// this is now also zero as well
						numLeftToAdd = 0;
					}
				}
				
				// if the number of blocks is reaching the end, change things
				if (numOutputsRemaining > 0) {
					if (numOutputsRemaining - numBlocks < 0) {
						numBlocks = numOutputsRemaining;
					}
					// artificialIntelligence::classes::calculatedAndUpdateLayerGPU<<< numBlocks, numThreads, sharedSize, stream1 >>>(current_input, current_weights, current_output, numBlocks, numOutputs, numPerThread, weightsAddedLastSet, startingOutputID);
					outputIndex += numBlocks;
					numOutputsRemaining -= numBlocks;
				}
				startingOutputID += numBlocks;
				
				
			} while (numOutputsRemaining > 0 || numLeftToAdd > 0);

			gpuErrchk(cudaDeviceSynchronize());
			// std::cout << "\n\n\n";
			weightsAddedLastSet = weightIndex;
		
			if (current_weights == device_weights1) {
				current_weights = device_weights2;
				next_weights = device_weights1;
			} else {
				current_weights = device_weights1;
				next_weights = device_weights2;
			}
			// std::cout << "startingOutputID: " << startingOutputID << '\n';
			// std::cout << "nextOutputID: " << nextOutputID << '\n';
			startingOutputID = nextOutputID;
			nextOutputID = numLastAdded;
			numOutputsRemaining = numOutputs;
			// std::cout << "startingOutputID: " << startingOutputID << '\n';
			// if on the last iteration for the layer, start doing the next layer

			// make sure the next layer of weights is done
		} while (!weightsFinished);
		
		// once its all done, then copy the memory over and prepare for next setup
		gpuErrchk(cudaMemcpy(output, current_output, numOutputs * sizeof(float), cudaMemcpyDeviceToHost));

		weightIndex = 0;
		
		// currentLayer->getLayerMatrix()->printMatrix();

		Matrix3D* bias = currentLayer->getBias();
		currentLayer = currentLayer->next;

		if (currentLayer->next != nullptr) {
			currentLayerMatrix = currentLayer->getLayerMatrix();
			output = currentLayer->next->getLayerMatrix()->getArr();
			numNodes = currentLayerMatrix->getSize() / sizeof(float); // the number of blocks that will be generated
			numOutputs = currentLayer->next->getLayerMatrix()->getSize() / sizeof(float);
			numWeights = numNodes * numOutputs;
			// std::cout << numWeights << '\n';

			// currentLayer->getLayerMatrix()->printMatrix();
		}
		*currentLayer->getLayerMatrix() += bias;
		sigmoid(currentLayer->getLayerMatrix(), false);

		gpuErrchk(cudaFree(current_input));
		gpuErrchk(cudaFree(current_output));
		gpuErrchk(cudaMalloc((void **) &current_input, currentLayerMatrix->getSize()));
		gpuErrchk(cudaMalloc((void **) &current_output, numOutputs * sizeof(float)));
		
	}




	// :::: FREE ALL ALLOCATED MEMORY :::: //
	// gpuErrchk(cudaFree(device_inputs1));	
	// gpuErrchk(cudaFree(device_output1));
	// gpuErrchk(cudaFree(device_weights1));	
	// gpuErrchk(cudaFree(device_weights2));

}

void BasicLayer::calculateAndUpdateAllGPUV2() {
	// make it so that the memcpy is not called a billion times, but rather once per time that the kernel runs, or even less than that.


	BasicLayer* currentLayer = this;
	Matrix3D* currentLayerMatrix = currentLayer->getLayerMatrix();
	// currentLayer->weights->print();

	long long numNodes = currentLayerMatrix->getSize() / sizeof(float); // the number of blocks that will be generated
	long long numOutputs = currentLayer->getNext()->getLayerMatrix()->getSize() / sizeof(float);
	long long numWeights = numNodes * numOutputs;
	long long numOutputsRemaining = numOutputs;
	long long numWeightsRemaining = numWeights;
	long long outputIndex = 0;

	// :::: TESTING SETTINGS :::: //

	// gpu kernel values
	// int numPerThread = 3;
	// int numBlocks = numOutputs / 2;
	// int numThreads = 1;

	// setting up starting values 
	// int maxWeightIndex = 15;

	// :::: REAL SETTINGS :::: //
	// int numPerThread = (int) log2 (numWeights) * 4; // making each node do log2(n) * 4 amount of work because each weight is one work and they run fairly fast
	long long numBlocks = numOutputs > 8192 ? 8192 : numOutputs; // break the number of blocks into chunks of 16384 or less
	long long numThreads = 512; // arbitrary
	long long maxWeightIndex = currentLayer->getWeights()->size;

	long long numPerThread = std::ceil ((double)maxWeightIndex / (numBlocks * numThreads)); // number of weights per iteration

	if (maxWeightIndex > numWeights) {
		maxWeightIndex = numWeights;
	}


	long long sharedSize = numThreads * sizeof(float); // sharedSize within the block should be same as number of threads, because each thread only uses one element in the shared dataset
	
	// input nodes and weights
	float* input = currentLayerMatrix->getArr();
	float* output = currentLayer->getNext()->getLayerMatrix()->getArr();

	// two sets so that speed can be increased 
	float* current_input;
	float* current_output;
	// streams for asynchronous
	cudaStream_t stream1, stream2;
	cudaStreamCreate ( &stream1); 
	cudaStreamCreate ( &stream2); 

	// two sets so that speed can be increased
	// only the weights are host because the others will be fast due to small sizes
		

	gpuErrchk(cudaMalloc((void **) &current_input, currentLayerMatrix->getSize()));
	gpuErrchk(cudaMalloc((void **) &current_output, numOutputs * sizeof(float)));

	// do first memcpy for weights
	long long matrixSize = currentLayer->getWeights()->weights->getSize() / sizeof(float);

	BasicWeight* currentWeight = currentLayer->getWeights();
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

	// std::cout << "numLeftToAdd" <<  numLeftToAdd << "\n";



	// :::: BEGIN INSERT FIRST BATCH OF WEIGHTS ASYNCHRONOUSLY :::: // 

	// set up the array for the cudaMallocPitch
	// the size of the weightArr will be the maxWeightIndex / number of weights per weight matrix

	// make the actual pitched array

	float* current_weights;
	float* next_weights;

	gpuErrchk(cudaMalloc((void **) &current_weights, maxWeightIndex * sizeof(float)));
	gpuErrchk(cudaMalloc((void **) &next_weights, maxWeightIndex * sizeof(float)));
	// currentLayerMatrix->printMatrix();
	gpuErrchk(cudaMemcpy(current_weights, currentWeight->getWeightMatrix(0)->getArr(), maxWeightIndex * sizeof(float), cudaMemcpyHostToDevice));
	weightsInCurrentKernelRun = maxWeightIndex;
	weightsAddedLastSet = maxWeightIndex;
	currentWeightMatrixIndex++;
	
	int startingOutputID = 0;
	int nextOutputID = maxWeightIndex % currentWeight->outputSize;
	int numWeightsMatrixesLeft = std::ceil((float)numWeights / maxWeightIndex) - 1;
	// std::cout << "\n\n\n";

	input = currentLayerMatrix->getArr();
	output = currentLayer->getNext()->getLayerMatrix()->getArr();
	gpuErrchk(cudaMemcpy(current_input, input, currentLayerMatrix->getSize(), cudaMemcpyHostToDevice)); // set the input layer to the input
	gpuErrchk(cudaMemset(current_output, 0b00000000, numOutputs * sizeof(float))); // set the output to zero

	int debugCounter = 0;
	// go through every single layer
   while (currentLayer->getNext() != nullptr) {
		currentWeightMatrixIndex = 1;
		numWeightsMatrixesLeft = std::ceil((float)numWeights / maxWeightIndex) - 1;
		outputIndex = 0;
		startingOutputID = 0;
		numOutputsRemaining = numOutputs;
		nextOutputID = weightsAddedLastSet;
		// copy the inputs and outputs to device
		bool weightsFinished = false;
		long long weightsUsed = 0;
		do {
			if (numWeightsMatrixesLeft >= 1){
				if (currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float) < maxWeightIndex) {
					maxWeightIndex = currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getSize() / sizeof(float);
				}
				weightsAddedLastSet = maxWeightIndex;
				// std::cout << "inside11\n";

				// std::cout << "currentWeightMatrixIndex: " << currentWeightMatrixIndex<< "\n";
				// std::cout << "maxWeightIndex: " << maxWeightIndex * sizeof(float)<< "\n";
				// if (currentWeightMatrixIndex == 3) {
				// 	if (next_weights == nullptr) {
				// 		std::cout << "nullptr1\n";
				// 	}
				// 	if (current_weights == nullptr) {
				// 		std::cout << "nullptr2\n";
				// 	}
				// 	if (currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr() == nullptr){
				// 		std::cout << "nullptr3\n";
				// 	}
				// 	exit(0);
				// }
				gpuErrchk(cudaMemcpyAsync(next_weights, currentWeight->getWeightMatrix(currentWeightMatrixIndex)->getArr(), maxWeightIndex * sizeof(float), cudaMemcpyHostToDevice));
				currentWeightMatrixIndex++;
				numWeightsMatrixesLeft -= 1;
				// std::cout << '\n';
			} else { 
				// std::cout << "inside12\n";
				// numWeights needs to be taken from the next values
				if (currentLayer->getNext()->getNext() != nullptr) {
					int nextNumWeights = numOutputs * currentLayer->getNext()->getNext()->getLayerMatrix()->getSize() / sizeof(float);
					int nextMaxWeightIndex = currentLayer->getNext()->getWeights()->getWeightMatrix(0)->getSize() / sizeof(float);
					if (nextMaxWeightIndex > nextNumWeights) {
						nextMaxWeightIndex = nextNumWeights;
					}
					gpuErrchk(cudaFree(next_weights));
					gpuErrchk(cudaMalloc((void **) &next_weights, nextMaxWeightIndex * sizeof(float)));
					// std::cout << "nextMaxWeightIndexBytes: " << nextMaxWeightIndex * sizeof(float)<< "\n";
					// std::cout << "nextMaxWeightIndex: " << currentLayer->getNext()->getWeights()->getWeightMatrix(0)->getSize() / sizeof(float) << "\n";
					
					// currentLayer->getNext()->getWeights()->getWeightMatrix(0)->printMatrix();
					gpuErrchk(cudaMemcpyAsync(next_weights, currentLayer->getNext()->getWeights()->getWeightMatrix(0)->getArr(), nextMaxWeightIndex * sizeof(float), cudaMemcpyHostToDevice));
					currentWeightMatrixIndex = 1;
					numWeightsMatrixesLeft = std::ceil((float)nextNumWeights / nextMaxWeightIndex) - 1;
					weightsAddedLastSet = nextMaxWeightIndex;
					// std::cout << '\n';
				}
				weightsFinished = true;
			}
			
			long long helper = 0;
			do {

				// if the number of blocks is reaching the end, change things
				if (numOutputsRemaining > 0) {
					// std::cout << "inside22\n";

					// std::cout << "numBlocks: " << numBlocks << '\n';
					// std::cout << "numOutputs: " << numOutputs << '\n';
					// std::cout << "numPerThread: " << numPerThread << '\n';
					// std::cout << "weightsInCurrentKernelRun: " << weightsInCurrentKernelRun << "\n\n";
					// std::cout << "numOutputsRemaining: " << numOutputsRemaining << '\n';
					// std::cout << "helper: " << helper << '\n';
					// std::cout << "weightsUsed: " << weightsUsed << "\n\n";
					// std::cout << "startingOutputID: " << startingOutputID << '\n';
					
					if (numOutputsRemaining - numBlocks < 0) {
						numBlocks = numOutputsRemaining;
					}

					artificialIntelligence::classes::calculatedAndUpdateLayerGPU<<< numBlocks, numThreads, sharedSize, stream1 >>>(current_input, current_weights, current_output, numBlocks, numOutputs, numPerThread, weightsInCurrentKernelRun, helper, weightsUsed, startingOutputID);
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
			numBlocks = numOutputs > 8192 ? 8192 : numOutputs;
			weightsInCurrentKernelRun = weightsAddedLastSet;

			// std::cout << "\n";
			float* temp = current_weights;
			current_weights = next_weights;
			next_weights = temp;

		} while (!weightsFinished);
		// once its all done, then copy the memory over and prepare for next setup
		gpuErrchk(cudaMemcpy(output, current_output, numOutputs * sizeof(float), cudaMemcpyDeviceToHost));

		weightIndex = 0;
		
		// currentLayer->getLayerMatrix()->printMatrix();
		// exit(0);
		Matrix3D* bias = currentLayer->getBias();
		currentLayer = currentLayer->getNext();
		currentLayerMatrix = currentLayer->getLayerMatrix();
		currentWeight = currentLayer->getWeights();
		numNodes = currentLayerMatrix->getSize() / sizeof(float);

		if (currentLayer->getNext() != nullptr) {
			output = currentLayer->getNext()->getLayerMatrix()->getArr();
			numOutputs = currentLayer->getNext()->getLayerMatrix()->getSize() / sizeof(float);
			numWeights = numNodes * numOutputs;
			maxWeightIndex = currentLayer->getWeights()->size;
			numBlocks = numOutputs > 8192 ? 8192 : numOutputs; // break the number of blocks into chunks of 16384 or less
			numThreads = 512; // arbitrary
			numPerThread = std::ceil ((double)maxWeightIndex / (numBlocks * numThreads));
			gpuErrchk(cudaFree(next_weights));
			gpuErrchk(cudaMalloc((void **) &next_weights, maxWeightIndex * sizeof(float)));
			// std::cout << "maxWeightIndex: " << maxWeightIndex << '\n';
			// std::cout << "currentLayerWeights: " << currentLayer->getWeights()->getWeightMatrix(0)->getSize() << "\n\n\n\n\n";
			// currentLayer->getLayerMatrix()->printMatrix();
			gpuErrchk(cudaFree(current_output));
			gpuErrchk(cudaMalloc((void **) &current_output, numOutputs * sizeof(float)));
			output = currentLayer->getNext()->getLayerMatrix()->getArr();
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
__global__ void artificialIntelligence::classes::calculatedAndUpdateLayerGPU(float* nodeValues, float* weights, float* output, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int outputNodeId = (blockIdx.x + startingOutputId) % outputSize;
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
   Matrix3D* weights;
   float activation = 0;

   // loop through every weight matrix
   // std::cout << "[" << this->layerMatrix->getLength() << "] " << "[" << this->layerMatrix->getWidth() << "] " << "[" << this->layerMatrix->getHeight() << "]   " 
   // << "[" << nextLayer->getLength() << "] " << "[" << nextLayer->getWidth() << "] " << "[" << nextLayer->getHeight() << "]" << "\n\n";
   for (int fl = 0; fl < this->layerMatrix->getLength(); fl++) {
      for (int fw = 0; fw < this->layerMatrix->getWidth(); fw++) {
         for (int fh = 0; fh < this->layerMatrix->getHeight(); fh++) {
            
            // making the activation start at the bias point
            // this returns the matrix for each node
            // now the matrix needs to be factored into each 
            // weights = this->weights->getWeightMatrix(fl, fw, fh);
         
            // std::cout << "[" << fl << "] " << "[" << fw << "] " << "[" << fh << "] " << '\n';

            // if (fw == 1) {
            //    if (weights == nullptr) {
            //       this->weights->print();
            //       layerMatrix->printMatrix();
            //       std::cout << "error";
            //       exit (0);
            //    }
            //    weights->printMatrix();
            // }

            for (int sl = 0; sl < nextLayer->getLength(); sl++) {
               for (int sw = 0; sw < nextLayer->getWidth(); sw++) {
                  for (int sh = 0; sh < nextLayer->getHeight(); sh++) {
                     
                     // if (isnan(activation)) {
                     //    std::cout << this->layerMatrix->getData(fl, fw, fh) << " ";
                     //    std::cout << weights->getData(sl, sw, sh) << " ";
                     //    std::cout << outputs->getData(sl, sw, sh) << " ";
                     //    outputs->printMatrix();
                     //    std::cout << "\n" << sl << " " << sw << " " << sh;
                     //    std::cout << "\nactivation\n";
                     //    exit (0);
                     // }
                     
                     activation = *this->layerMatrix->getData(fl, fw, fh) * *this->weights->getData(fl, fw, fh, sl, sw, sh) + *outputs->getData(sl, sw, sh);

                     // std::cout << "[" << fl << "] " << "[" << fw << "] " << "[" << fh << "]   " << "[" << sl << "] " << "[" << sw << "] " << "[" << sh << "]" << '\n';
                        
                     // std::cout << *this->layerMatrix->getData(fl, fw, fh) << "    " << *weights->getData(sl, sw, sh) << "   " << activation <<  '\n';
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
   // exit (0);
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
