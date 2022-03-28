#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <sys/resource.h>

#include <artificialIntelligence/basicLearningTypes/generationalAIBasic.hpp>
#include <artificialIntelligence/basicLearningTypes/generationalAIBasic.cuh>
#include <artificialIntelligence/functions/activationFunctions.cuh>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include <coreutils/util/cudaErrors.cuh>

#include <coreutils/functions/math/simpleMath.hpp>
#include <coreutils/functions/sort/sortHelpers.cpp>
#include <coreutils/functions/debug/print.cpp>

#include <artificialIntelligence/classes/BasicLayer.cuh>
#include <artificialIntelligence/classes/BasicLayerList.hpp>

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

         void run (BasicLayerList* list, int epochs, double learningRate, Matrix3D** inputDataMatrixes, Matrix3D** outputDataMatrixes, int inputCount, bool calculateError, bool print) {

            // initial error

            std::cout.precision(4);

            double sumInitial = 0;
            // if (calculateError) {
				// 	std::cout << "Calculating Initial Error.\n";
            //    for (int i = 0; i < inputCount; i++) {
            //       list->editRootMatrix(inputDataMatrixes[i]);
				// 		if (GPU) { list->calculateAndUpdateAllGPUV2(); }
				// 		else { list->calculateAndUpdateAllCPU(); }
            //       Matrix3D* error = *outputDataMatrixes[i] - list->getLast()->getLayer();
            //       Matrix3D* squared = *error * error;
            //       sumInitial += squared->sum() * 100;
            //       delete error;
            //       delete squared;
            //    }
            //    std::cout << "Total initial error :: " << sumInitial << "%\n\n";
            // }
				
            int* order = new int[inputCount];
            for (int i = 0; i < inputCount; i++) {
               order[i] = i;
            }
   
            // main loop
				float initialError = 0;
            std::cout << std::fixed;
				std::cout.precision(2);
            for (int e = 0; e < epochs; e++) {
               // because stochastic gradient descent, the order needs randomization

					// debug::printArr(order, 10);
               sort::shuffle(order, inputCount);
               // debug::printArr(order, 10);
                
               if (calculateError) {
                  float currentError = 0;
						std::cout << "Calculating error.\n";
                  for (int i = 0; i < inputCount; i++) {
                     list->editRootMatrix(inputDataMatrixes[i]);
							if (GPU) { list->calculateAndUpdateAllGPUV2(); }
							else { list->calculateAndUpdateAllCPU(); }
							Matrix3D* error = *outputDataMatrixes[i] - list->getLast()->getLayer();
							Matrix3D* squared = *error * error;
							currentError += squared->sum() * 100;
							delete error;
							delete squared;
						}
						if (initialError < 1) {
							initialError = currentError;
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

               for (int i = 0; i < inputCount; i++) {
                  // update the list with random input
                  list->editRootMatrix(inputDataMatrixes[order[i]]);
                  if (GPU) { list->calculateAndUpdateAllGPUV2(); }
						else { list->calculateAndUpdateAllCPU(); }

						if (inputCount < 100 || i % (inputCount / 100) == 0) {
							printf("%2.2f", e / (double) epochs * 100 + ((float) i / inputCount) / epochs * 100);
							std::cout << "%\n";
						}

						// exit(0);
                  // std::cout << "i: " << i << "   order[i]: " << order[i] << '\n';
                  // backpropagation starts at root
						
                  BasicLayer* currentLayer = list->getLast();

                  // do math for deltaOutput
                  Matrix3D* currentLayerMatrix = currentLayer->getLayer();
                  Matrix3D* error = *(outputDataMatrixes[order[i]]) - currentLayerMatrix;
						// outputDataMatrixes[order[i]]->printMatrix();
						
                  Matrix3D* dSig = dSigmoid (currentLayerMatrix);
                  Matrix3D* deltaNext = *error * (dSig);
                  Matrix3D* deltaPrev = new Matrix3D (deltaNext->getLength(), deltaNext->getWidth(), deltaNext->getHeight());
					

                  delete error;
                  delete dSig;

                  deltaPrev->setMatrix(deltaNext);

                  // calculate and set the bias
                  currentLayer = currentLayer->getPrev();
                  
                  int counter = 0;


                  while (currentLayer->getPrev() != nullptr) {
							
                     // counter for debuggin
                     counter++;
							
                     // get the layerMatrix
                     currentLayerMatrix = currentLayer->getLayer();

							int currentLength = currentLayerMatrix->getLength();
							int currentWidth = currentLayerMatrix->getWidth();
							int currentHeight = currentLayerMatrix->getHeight();
                     // currentLayerMatrix->printMatrix();
                     delete deltaPrev;
                     deltaPrev = new Matrix3D (deltaNext->getLength(), deltaNext->getWidth(), deltaNext->getHeight());
                     deltaPrev->setMatrix(deltaNext);
							
							// error = new Matrix3D (currentLength, currentWidth, currentHeight);
							
							// calculates the impact of each node and puts it into the weighted matrix. This is used to calculate error

							if (GPU) { error = currentLayer->calculateErrorGPU(deltaPrev); }
							else { error = currentLayer->calculateErrorCPU(deltaPrev); }

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
							// BasicLayer *tempLayer = new BasicLayer(*currentLayer, true);
							// tempLayer->updateWeightsCPU(deltaPrev, learningRate);
							if (GPU) { currentLayer->updateWeightsGPU(deltaPrev, learningRate); }
							else { currentLayer->updateWeightsCPU(deltaPrev, learningRate); }
							// tempLayer->print(1,1);
							// currentLayer->print(1,1);
							// exit(0);
	
                     currentLayer = currentLayer->getPrev();
                  }
						
                  delete deltaNext;
                  delete deltaPrev;

						// std::cout << "here1\n";
						// gpuErrchk(cudaDeviceReset());
						// std::cout << "here2\n";
               }

               // list->getLast()->print();
               if (isnan (*list->getLast()->getLayer()->getData(0,0,0))) {
						if (print) list->print(1,1);
                  std::cout << "here2\n";
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
                  list->editRootMatrix(inputDataMatrixes[i]);
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

            delete[] order;
         }
      }
   }
}