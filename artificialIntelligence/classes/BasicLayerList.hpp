#ifndef BASIC_LAYER_LIST_HPP
#define BASIC_LAYER_LIST_HPP

#include <iostream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include "BasicLayer.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

namespace artificialIntelligence {
   namespace classes {

      
      class BasicLayerList {
         public:
            BasicLayerList (Matrix3D* layer, Matrix3D* biasMatrix = nullptr, BasicWeight* weights = nullptr);
            
            BasicLayerList ();

				~BasicLayerList ();

				BasicLayerList(const BasicLayerList& bll);

            void print (bool printBias = false, bool printWeights = false);

            void add (BasicLayer* layer);

            void add (Matrix3D* layer, Matrix3D* biasMatrix = nullptr, BasicWeight* weights = nullptr);

            void addNew (int length, int width, int height);

				void calculateAndUpdateAllGPU ();

				void calculateAndUpdateAllGPUV2 ();
				
            void calculateAndUpdateAllCPU ();

            void calculateAndUpdateLastCPU ();

            void editRootMatrix (Matrix3D* newMatrix);

            BasicLayer* getRoot ();

            BasicLayer* getLast ();

            void toFile (std::string filepath);

            static BasicLayerList* loadFromFile (std::string filepath);

         private:
            BasicLayer* root;
            BasicLayer* last;

      };
   }
}


#endif