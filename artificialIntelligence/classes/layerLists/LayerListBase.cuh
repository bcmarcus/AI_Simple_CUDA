#ifndef LAYER_LIST_TEMPLATE
#define LAYER_LIST_TEMPLATE

#include <iostream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "LayerListBase.cuh"
#include "../layers/LayerBase.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

namespace artificialIntelligence {
   namespace classes {

      
      class LayerListBase {
			protected:
            enum ListClass {
               Default,
               Basic
            };

            // first layer
            LayerBase* root;

            // last layer
            LayerBase* last;

            // the type of class
            ListClass type;

            // number of params
            long long totalParamCount;
			public: 
            // -- GET METHODS -- //

            // gets the root layer
            LayerBase* getRoot () {
               return this->root;
            }

            // gets the last layer
            LayerBase* getLast () {
               return this->last;
            }


            // -- SET METHODS -- //
            
            // sets the root matrix
            void setRootMatrix (Matrix3D* newMatrix) {
               if (this->root != nullptr) {
                  this->root->setLayerMatrix(newMatrix);
               }
            }


            // -- GENERATE METHODS -- //

            // adds a layer at the end of the model recursively 
            void add (LayerBase* layer, int index = 0) {
               if (this->last != nullptr) {
                  this->last = this->last->add(layer, index);
                  totalParamCount += this->last->getPrev(0)->paramCount();
               } else {
                  this->root = layer;
                  this->last = this->root;
               }
            }

            // creates and adds a layer at the end of the model recursively
            // void add (Matrix3D* layer, Matrix3D* biasMatrix, WeightBase* weights);


            // -- LAYER UPDATE METHODS -- //
            
            // updates all layers in the model using CPU compute
            void calculateAndUpdateAllCPU () {
               if (this->root != nullptr) {
                  // switch (this->type){
                  //    case this->ListClass::Default:
                  //    case this->ListClass::Basic:

                  // }
                  ((LayerBase*) (this->root))->calculateAndUpdateAllCPU();
               } else {
                  std::cout << "No root layer initialized!\n";
               }
            }

            // updates the last layer using CPU compute
            void calculateAndUpdateLastCPU ();

            // updates all layers in the model using GPU compute revised
				void calculateAndUpdateAllGPUV2 ();


            // -- PRINT METHODS -- //
            
            // prints the entire model
            void print (bool printLayer = false, bool printBias = false, bool printWeights = false);


            // -- LOAD FILE METHODS -- //

            // loads a model into a file using the format of 
            // layer length, layer width, layer height
            // bias length, bias width, bias height
            // <the values for the bias, all comma seperated>
            // layer length, layer width, layer height, bias length, bias width, bias height
            // <the values for the weights, with each float16 represented by 4 bytes of data> 
            void toFile (std::string filepath);
		};
	}
}

#endif