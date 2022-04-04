#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <sys/resource.h>

// #include <artificialIntelligence/basicLearningTypes/generationalAIBasic.hpp>
#include <artificialIntelligence/basicLearningTypes/generationalAIBasic.cuh>
#include <artificialIntelligence/functions/activationFunctions.cuh>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include <coreutils/util/cudaErrors.cuh>

#include <coreutils/functions/math/simpleMath.hpp>
#include <coreutils/functions/sort/sortHelpers.hpp>
#include <coreutils/functions/debug/print.hpp>

#include <artificialIntelligence/classes/layers/BasicLayer.cuh>
#include <artificialIntelligence/classes/layerLists/BasicLayerList.cuh>

#include <device_launch_parameters.h>

#define GPU 1

using namespace coreutils::classes::matrixes;
using namespace coreutils::functions;
using namespace artificialIntelligence::classes;
using namespace artificialIntelligence::functions::activation;
using namespace std;

namespace artificialIntelligence {
   namespace basicLearningTypes {
      namespace generationalAIBasic {

         void runStochasticGradientDescent (BasicLayerList* list, int epochs, double learningRate, Matrix3D** inputDataMatrixes, Matrix3D** outputDataMatrixes, int inputCount, int batchSize, bool calculateError, bool print) {

            // set printing values
            std::cout << std::fixed;
				std::cout.precision(2);

				// initial error within the model
				float initialError = 0;

				// no batches if the size is not set
				if (batchSize == 0) {
					batchSize = inputCount;
				}

            // generate the random order that the data will be chosen in
            int* order = new int[inputCount];
            for (int i = 0; i < inputCount; i++) {
               order[i] = i;
            }

            // loop through epochs
            for (int e = 0; e < epochs; e++) {
					
               // because stochastic gradient descent, the order needs randomization
               sort::shuffle(order, inputCount);
               
               // calculates the first error
               if (calculateError) {
                  float currentError = 0;
						std::cout << "Calculating error.\n";

                  // run through all of the data and calculate the error for it
                  for (int i = 0; i < inputCount; i++) {
                     list->setRootMatrix(inputDataMatrixes[i]);
							if (GPU) { list->calculateAndUpdateAllGPUV2(); }
							else { list->calculateAndUpdateAllCPU(); }
							Matrix3D* error = *outputDataMatrixes[i] - list->getLast()->getLayer();
							Matrix3D* squared = *error * error;
							currentError += squared->sum() * 100;
							delete error;
							delete squared;
						}

                  // set the initial error if this is the first one
						if (initialError < 1) {
							initialError = currentError;
						}
                  std::cout << "Total error :: " << currentError << "%\n\n";
               }

               // calculate the error for each data point and correct it with backpropagation
					for (int k = 0; k < inputCount / batchSize; k++) {
						for (int i = 0; i < batchSize; i++) {

							// initialize the first layer to the data point
							list->setRootMatrix(inputDataMatrixes[order[i + k * batchSize]]);

							// calculate the output layer with the given data point
							if (GPU) { list->calculateAndUpdateAllGPUV2(); }
							else { list->calculateAndUpdateAllCPU(); }

							// tells the user how far along the training is going
							if (inputCount < 100 || i % (inputCount / 100) == 0) {
								printf("%2.2f", e / (double) epochs * 100 + (((float) i + k * batchSize) / inputCount) / epochs * 100);
								std::cout << "%\n";
							}
							

							// -- STARTING BACKPROPAGATION -- //

							// gets the final layer
							BasicLayer* currentLayer = list->getLast();

							// gets difference in the final calculated layer and what the final output layer should be.
							Matrix3D* currentLayerMatrix = currentLayer->getLayer();
							Matrix3D* error = *(outputDataMatrixes[order[i + k * batchSize]]) - currentLayerMatrix;

							// calculates the derivate of the sigmoid function and multiplies by error for the 
							Matrix3D* dSig = dSigmoid (currentLayerMatrix);
							Matrix3D* deltaNext = *error * (dSig);
							Matrix3D* deltaPrev = new Matrix3D (deltaNext->getLength(), deltaNext->getWidth(), deltaNext->getHeight());
						

							delete error;
							delete dSig;

							deltaPrev->setMatrix(deltaNext);

							// calculate and set the bias
							currentLayer = currentLayer->getPrev();
							
							while (currentLayer->getPrev() != nullptr) {

								// get the layerMatrix
								currentLayerMatrix = currentLayer->getLayer();

								// free previous delta memory and set the new one
								delete deltaPrev;
								deltaPrev = new Matrix3D (deltaNext->getLength(), deltaNext->getWidth(), deltaNext->getHeight());
								deltaPrev->setMatrix(deltaNext);

								// gets the error for the current layer
								if (GPU) { error = currentLayer->calculateErrorGPU(deltaPrev); }
								else { error = currentLayer->calculateErrorCPU(deltaPrev); }

								// frees the next delta and creates the next one
								delete deltaNext;
								dSig = dSigmoid (currentLayerMatrix);
								deltaNext = *error * (dSig);

								// free more memory
								delete error;
								delete dSig;

								// calculate the bias and set it for this node
								Matrix3D* bias = *deltaPrev * learningRate; 
								currentLayer->setBias(bias);
								delete bias;

								// update the weights for this layer
								if (GPU) { currentLayer->updateWeightsGPU(deltaPrev, learningRate); }
								else { currentLayer->updateWeightsCPU(deltaPrev, learningRate); }
		
								// go to the previous layer for more back propagation
								currentLayer = currentLayer->getPrev();
							}
							
							// free memory
							delete deltaNext;
							delete deltaPrev;
						}

						// print if error
						if (isnan (*list->getLast()->getLayer()->getData(0,0,0))) {
							if (print) list->print(1,1);
							std::cout << "here2\n";
							exit (0);
						}
					}
            }

				// printing
            if (print) {
               list->print(true, true);
            }

				// printing to see if all of them work properly
            if (print) {
               for (int i = 0; i < inputCount; i++) {
                  std::cout << "\n\n\n\n";
                  std::cout << "Input Matrix: ";
                  inputDataMatrixes[i]->printMatrix();
                  std::cout << "True Output: ";
                  outputDataMatrixes[i]->printMatrix();
                  list->setRootMatrix(inputDataMatrixes[i]);

                  if (GPU) { list->calculateAndUpdateAllGPUV2(); }
						else { list->calculateAndUpdateAllCPU(); }
						
                  std::cout << "Calculated Output: ";
                  list->getLast()->getLayer()->printMatrix();
                  std::cout << "\n\n";
               }
            }

            // final error
            std::cout.precision(4);
            if (calculateError) {
               double sumFinal = 0;
					std::cout << "Calculating final error\n";
               for (int i = 0; i < inputCount; i++) {
                  list->setRootMatrix(inputDataMatrixes[i]);
                  if (GPU) { list->calculateAndUpdateAllGPUV2(); }
						else { list->calculateAndUpdateAllCPU(); }
                  Matrix3D* error = *outputDataMatrixes[i] - list->getLast()->getLayer();
                  Matrix3D* squared = *error * error;
                  sumFinal += squared->sum() * 100;
                  delete error;
                  delete squared;
               }
					std::cout << "Total initial error :: " << initialError << "%\n";
               std::cout << "Total final error :: " << sumFinal << "%\n";
            }

				// final cleanup
            delete[] order;
         }
      }
   }
}