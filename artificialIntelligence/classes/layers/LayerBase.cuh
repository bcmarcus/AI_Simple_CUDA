#ifndef LAYER_TEMPLATE
#define LAYER_TEMPLATE

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include "../../functions/activationFunctions.cuh"

#include "../weights/WeightBase.cuh"
#include "../weights/BasicWeight.cuh"
#include "../weights/ConvWeight.cuh"
#include "../weights/PoolWeight.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;
using namespace artificialIntelligence::functions::activation;

namespace artificialIntelligence {
   namespace classes {
		class LayerBase {
         public:
            enum LayerType {
               Basic,
               Conv,
               Pool
            };

         protected:
				// layer values
            Matrix3D** layerMatrixes;

				// next biases
            Matrix3D** biasMatrixes;

				// next weights
            WeightBase** weights;

            // next layers
            LayerBase** next;

            // previous layers
            LayerBase** prev;

            int layerMatrixCount;
            int biasCount;
            int weightsCount;
            int nextCount;
            int prevCount;

            ActivationType activationType;
            LayerType type;
			public:

            int getLayerMatrixCount () const {
					return this->layerMatrixCount;
				}

            int getBiasCount () const {
					return this->biasCount;
				}

            int getWeightsCount () const {
					return this->weightsCount;
				}

            int getNextCount () const {
					return this->nextCount;
				}

            int getPrevCount () const {
					return this->prevCount;
				}

            ActivationType getActivationType () const {
					return this->activationType;
				}

            LayerType getLayerType () const {
					return this->type;
				}

            LayerBase* getNext (int index = 0) const {
               if (index >= this->nextCount) {
                  return nullptr;
               }
               if (this->next == nullptr) {
                  return nullptr;
               }
					return this->next[index];
				}

				LayerBase* getPrev (int index = 0) const {
               if (index >= this->prevCount) {
                  return nullptr;
               }
               if (this->prev == nullptr) {
                  return nullptr;
               }
					return this->prev[index];
				}

				LayerBase* getLast () {
               if (this->getNext(0) == nullptr) {
                  return this;
               }
					return this->getNext(0)->getLast();
				}

				Matrix3D* getLayerMatrix (int index = 0) const {
               if (index >= this->layerMatrixCount) {
                  return nullptr;
               }
               if (this->layerMatrixes == nullptr) {
                  return nullptr;
               }
					return this->layerMatrixes[index];
				}

				Matrix3D* getBias (int index = 0) const {
               if (index >= this->biasCount) {
                  return nullptr;
               }
               if (this->biasMatrixes == nullptr) {
                  return nullptr;
               }
					return this->biasMatrixes[index];
				}

				WeightBase* getWeights (int index = 0) const {
               if (index >= this->weightsCount) {
                  return nullptr;
               }
               if (this->weights == nullptr) {
                  return nullptr;
               }
					return this->weights[index];
				}

            ActivationType getActivationType () {
               return this->activationType;
            }

            long long paramCount () {
               long long params = 0;
               for (int i = 0; i < this->weightsCount; i++) {
                  params += this->getWeights(i)->paramCount();
               }
               return params;
            }

            // -- SET METHODS -- //

            void copyLayerMatrix (Matrix3D* layerMatrix, int index = 0) {
               if (this->getLayerMatrix(index) == nullptr) {
						this->layerMatrixes[index] = new Matrix3D (layerMatrix->getLength(), layerMatrix->getWidth(), layerMatrix->getHeight());
                  this->layerMatrixCount++;
					}
					this->layerMatrixes[index]->setMatrix(layerMatrix);
            }

            // sets the current layer matrix
            void setLayerMatrix (Matrix3D* layerMatrix, int index = 0) {
               if (this->getLayerMatrix(index) != nullptr) {
                  delete this->getLayerMatrix(index);
                  this->layerMatrixCount--;
               }
					this->layerMatrixes[index] = layerMatrix;
               this->layerMatrixCount++;
				}

            void setNext (LayerBase* next, int index = 0) {
               if (this->getNext(index) != nullptr) {
                  delete this->getNext(index);
                  this->nextCount--;
               }
               this->next[index] = next;
               this->nextCount++;
            } 

            void setPrev (LayerBase* prev, int index = 0) {
               if (this->getPrev(index) != nullptr) {
                  delete this->getPrev(index);
                  this->prevCount--;
               }
               this->prev[index] = prev;
               this->prevCount++;
            } 

            void setBias (Matrix3D* biasMatrix, int index = 0) {
               if (this->getBias (index) != nullptr) {
                  delete this->getBias (index);
                  this->biasCount--;
               }
               this->biasMatrixes[index] = biasMatrix;
               this->biasCount++;
            }

            void setWeights (WeightBase* weights, int index = 0) {
               if (this->getWeights(index) != nullptr) {
                  delete this->getWeights(index);
                  this->weightsCount--;
               }
               this->weights[index] = weights;
               this->weightsCount++;
            }

            void setActivation (ActivationType type) {
               this->activationType = type;
            }

            // -- GENERATE METHODS -- // 

            // adds a layer at the end of the model recursively
            LayerBase* add (LayerBase* layer, int index) {
               if (this->getNext(index) == nullptr) {
                  this->next[index] = layer;
                  this->nextCount++;

                  this->next[index]->prev[0] = this;
                  this->next[index]->prevCount++;

                  if (this->biasMatrixes[0] == nullptr) {
                     this->biasMatrixes[0] = this->newBias (index);
                     this->biasCount++;
                  }

                  if (this->weights[0] == nullptr) {
                     this->weights[0] = this->newWeight (index);
                     this->weightsCount++;
                  }
               } 
               
               else {
                  this->next[0] = this->getNext()->add(layer, index);
                  this->getNext(0)->prev[0] = this;
               }

               return this;
            }

            // creates and adds a layer at the end of the model recursively
            // LayerBase* add (Matrix3D* layer, Matrix3D* biasMatrix = nullptr, WeightBase* weights = nullptr);

            // creates a new weight based on the two given layers
				virtual WeightBase* newWeight (int index) = 0;
            
            // creates a new bias based on the two given layers
            virtual Matrix3D* newBias (int index) = 0;


            // -- LAYER UPDATE METHODS -- //

            // updates all layers in the model using CPU compute
            virtual void calculateAndUpdateAllCPU () = 0;

            // updates this layer using CPU compute
            virtual void calculateAndUpdateLayerCPU () = 0;

            // updates all layers in the model using GPU compute revised
				virtual void calculateAndUpdateAllGPUV2 () = 0;

            // updates this layer using GPU compute
			 	// virtual void calculateAndUpdateLayerGPU();

            // calculates the error in the layer using CPU compute
				virtual Matrix3D* calculateErrorCPU (Matrix3D* delta) = 0;

            // calculates the error in the layer using GPU compute
				virtual Matrix3D* calculateErrorGPU (Matrix3D* delta) = 0;

            // updates weights in the layer using CPU compute
				virtual void updateWeightsCPU(Matrix3D* delta, double learningRate) = 0;

            // updates weights in the layer using GPU compute
				virtual void updateWeightsGPU (Matrix3D* delta, double learningRate) = 0;


            // -- PRINT METHODS -- //
            virtual void printDetails () = 0;

            // prints the layer and all layers below it
            void print (bool printLayer = false, bool printBias = false, bool printWeights = false, int index = 0, int depth = 1) {
               if (this->layerMatrixes != nullptr) {
                  std::cout << "\n\nDepth::Across " << depth << "::" << index << '\n';
                  this->printDetails();

                  if (this->weightsCount != 0) {
                     std::string a;
                     switch (this->activationType) {
                        case ActivationType::Sigmoid:
                           a = "Sigmoid";
                           break;
                        case ActivationType::Tanh:
                           a = "Tanh";
                           break;
                        case ActivationType::Relu:
                           a = "Relu";
                           break;
                        case ActivationType::LeakyRelu:
                           a = "LeakyRelu";
                           break;
                     }
                     std::cout << "Activation: " << a << '\n';
                     std::cout << "Parameter Count: " << paramCount() << '\n';
                  } else {
                     std::cout << "Activation: None" << '\n';
                     std::cout << "No Parameters.\n";
                  }
               } else {
                  std::cout << "No layer found!\n";
                  return;
               }

               if (printLayer) {
                  if (this->layerMatrixCount != 0) {
                     std::cout << "Layer Matrixes: \n";
                     for (int i = 0; i < this->layerMatrixCount; i++) {
                        this->getLayerMatrix(i)->printMatrix();
                     }
                  }
               }

               if (printBias) {
                  if (this->biasCount != 0) {
                     std::cout << "Bias Matrixes: \n";
                     for (int i = 0; i < this->biasCount; i++) {
                        this->getBias(i)->printMatrix();
                     }
                  } else {
                     std::cout << "No biases found!\n";
                  }
               }

               if (printWeights) {
                  if (this->weightsCount != 0) {
                     std::cout << "Weight Matrixes: \n";
                     for (int i = 0; i < this->weightsCount; i++) {
                        this->getWeights(i)->print();
                     }
                  } else {
                     std::cout << "No weights found!\n";
                  }
               }

               for (int i = 0; i < this->nextCount; i++) {
                  this->getNext(i)->print(printLayer, printBias, printWeights, i, depth + 1);
               }
            }


            // -- LOAD AND UNLOAD FILE METHODS -- //

            // loads a model into a file using the format of 
            // layer length, layer width, layer height
            // bias length, bias width, bias height
            // <the values for the bias, all comma seperated>
            // layer length, layer width, layer height, bias length, bias width, bias height
            // <the values for the weights, with each float16 represented by 4 bytes of data> 
            virtual void toFile (std::ofstream* outputFile) = 0;
		};
	}
}

#endif