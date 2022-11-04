#ifndef Conv_LAYER_LIST_HPP
#define Conv_LAYER_LIST_HPP

#include <iostream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "../layerLists/LayerListBase.cuh"
#include "../layers/LayerBase.cuh"
#include "../weights/WeightBase.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

namespace artificialIntelligence {
   namespace classes {
      class LayerList : public LayerListBase {
         public:
            // -- CONSTRUCTOR DESTRUCTOR COPY -- //

            // default constructor
				LayerList ();
							
            LayerList (LayerBase* layer);
				// constructor
            // LayerList (Matrix3D* layer, Matrix3D* biasMatrix = nullptr, WeightBase* weights = nullptr);
            
				// loads a model from a file
				LayerList (std::string filepath);

            // destructor
				~LayerList ();

            // copy constructor
				// LayerList(const LayerList& cll);

            
            // -- GET METHODS -- //

            // gets the root layer
            LayerBase* getRoot ();

            // gets the last layer
            LayerBase* getLast ();


            // -- SET METHODS -- //
            
            // sets the root matrix
            void copyRootMatrix (Matrix3D* newMatrix);


            // -- GENERATE METHODS -- //

            // adds a layer at the end of the model recursively 
            void add (LayerBase* layer, int index = 0);

            // creates and adds a layer at the end of the model recursively
            void add (Matrix3D* layer, Matrix3D* biasMatrix, WeightBase* weights);

            // creates and adds a new basic layer
            void addNewBasic (int length, int width, int height, ActivationType activationType = ActivationType::Sigmoid, int index = 0);

            // creates and adds a new pooling layer
            void addNewPool (int poolLength = 1, int poolWidth = 2, int poolHeight = 2, ActivationType activationType = ActivationType::Sigmoid, int index = 0);

            // creates and adds a layer at the end of the model recursively
            void addNewConv (int length, int width, int height, int convLength = 1, int convWidth = 3, int convHeight = 3, int features = 1, int stride = 1, ActivationType activationType = ActivationType::Sigmoid, int index = 0);
            

            // -- LAYER UPDATE METHODS -- //
            
            // updates all layers in the model using CPU compute
            void calculateAndUpdateAllCPU ();

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

				// loads a model from a file using the format described above
				static LayerList* loadFromFile (std::string filepath);
      };
   }
}


#endif