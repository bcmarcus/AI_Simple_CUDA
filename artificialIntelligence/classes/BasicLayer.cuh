#ifndef BASIC_LAYER_HPP
#define BASIC_LAYER_HPP

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <artificialIntelligence/classes/BasicWeight.cuh>

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

namespace artificialIntelligence {
   namespace classes {
		__global__ void calculatedAndUpdateLayerGPU(float* nodeValues, float* weights, float* output, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId);

      class BasicLayer{
         private:
            Matrix3D* layerMatrix;
            Matrix3D* biasMatrix;
            BasicWeight* weights;
            BasicLayer* next;
            BasicLayer* prev;

         public:
            BasicLayer (Matrix3D* layerMatrix, Matrix3D* biasMatrix = nullptr, BasicWeight* weights = nullptr);

            BasicLayer (int length, int width, int height);

            BasicLayer ();

            ~BasicLayer ();

            int print (bool printBias = false, bool printWeights = false, int depth = 1);

            BasicLayer* add (BasicLayer* layer);

            BasicLayer* add (Matrix3D* layer, Matrix3D* biasMatrix = nullptr, BasicWeight* weights = nullptr);

            void calculateAndUpdateAllCPU ();

				void calculateAndUpdateAllGPU ();

				void calculateAndUpdateAllGPUV2 ();

			 	void calculateAndUpdateLayerGPU();

            void calculateAndUpdateLayerCPU ();

            void setPrev (BasicLayer* prev);

            Matrix3D* getLayerMatrix ();

            void setLayerMatrix (Matrix3D* layerMatrix);

				BasicWeight* getWeights ();

				BasicWeight* newWeight (BasicLayer* firstLayer, BasicLayer* secondLayer);

            Matrix3D* getBias ();

            void setBiasMatrix (Matrix3D* bias);

            BasicLayer* getLast ();

            BasicLayer* getNext ();

            BasicLayer* getPrev ();

            void toFile (std::ofstream* outputFile);
            
            static BasicLayer* loadFromFile (std::ifstream* inputFile, BasicLayer* prev = nullptr);
      };
		
   }
}
#endif