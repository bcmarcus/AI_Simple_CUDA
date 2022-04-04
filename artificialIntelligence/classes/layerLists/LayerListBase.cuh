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
            // first layer
            LayerBase* root;

            // last layer
            LayerBase* last;

			public: 
				// -- GET METHODS -- //

				// gets the root layer
				LayerBase* getRoot ();

				// gets the last layer
				LayerBase* getLast ();

				// -- SET METHODS -- //
					
				// sets the root matrix
				void setRootMatrix (Matrix3D* newMatrix);


				// -- PRINT METHODS -- //

				// prints the entire model
				void print (bool printBias = false, bool printWeights = false);


				// -- LOAD AND UNLOAD FILE METHODS -- //

				// loads a model into a file
				void toFile (std::string filepath);

				// loads a model from a file using the format described above
				LayerListBase* loadFromFile (std::string filepath);

		};
	}
}

#endif