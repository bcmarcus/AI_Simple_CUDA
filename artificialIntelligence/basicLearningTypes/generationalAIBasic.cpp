#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <sys/resource.h>

#include <coreutils/classes/matrixes/Matrix3D.cpp>
#include <artificialIntelligence/basicLearningTypes/generationalAIBasic.hpp>
#include <artificialIntelligence/functions/activationFunctions.hpp>

#include <coreutils/functions/math/simpleMath.hpp>
#include <coreutils/functions/sort/sortHelpers.cpp>
#include <coreutils/functions/debug/print.cpp>

#include <artificialIntelligence/classes/BasicLayer.hpp>
#include <artificialIntelligence/classes/BasicLayerList.hpp>

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
                     // currentLayerMatrix->printMatrix();
                     delete deltaPrev;
                     deltaPrev = new Matrix3D (deltaNext->getLength(), deltaNext->getWidth(), deltaNext->getHeight());
                     deltaPrev->setMatrix(deltaNext);
                   
							error = new Matrix3D (currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
                     for (int l = 0; l < currentLayerMatrix->getLength(); l++) {
                        for (int w = 0; w < currentLayerMatrix->getWidth(); w++) {
                           for (int h = 0; h < currentLayerMatrix->getHeight(); h++) {
                              // currentLayer->print(true, true);
                              Matrix3D* weightedMatrix = *currentLayer->getWeights(l, w, h) * deltaPrev;
                                 
                              error->insert(weightedMatrix->sum(), l, w, h);
                              delete weightedMatrix;
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
                     for (int l = 0; l < currentLayerMatrix->getLength(); l++) {
                        for (int w = 0; w < currentLayerMatrix->getWidth(); w++) {
                           for (int h = 0; h < currentLayerMatrix->getHeight(); h++) {
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