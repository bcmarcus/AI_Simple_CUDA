#ifndef BASIC_LAYER_HPP
#define BASIC_LAYER_HPP

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include "BasicWeight.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

namespace artificialIntelligence {
   namespace classes {
		__global__ void calculateAndUpdateLayerGPU(float* nodeValues, float* weights, float* output, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId);

		__global__ void calculateError(float* weights, float* delta, float* error, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId);

		__global__ void updateWeights(float* weights, float* delta, float* input, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId, double learningRate);

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

				BasicLayer (const BasicLayer& b, bool copyAll = false);

            int print (bool printBias = false, bool printWeights = false, int depth = 1);

            BasicLayer* add (BasicLayer* layer);

            BasicLayer* add (Matrix3D* layer, Matrix3D* biasMatrix = nullptr, BasicWeight* weights = nullptr);
	
            void calculateAndUpdateAllCPU ();

				void calculateAndUpdateAllGPU ();

				void calculateAndUpdateAllGPUV2 ();

			 	void calculateAndUpdateLayerGPU();

            void calculateAndUpdateLayerCPU ();

            void setPrev (BasicLayer* prev);

            Matrix3D* getLayer () const;

            void setLayer (Matrix3D* layerMatrix);

				void setBias (Matrix3D* biasMatrix);

				void setWeights (BasicWeight* weight);

				BasicWeight* getWeights () const;

				BasicWeight* newWeight (BasicLayer* firstLayer, BasicLayer* secondLayer);

            Matrix3D* getBias () const;

            void setBiasMatrix (Matrix3D* bias);

            BasicLayer* getLast ();

            BasicLayer* getNext () const;

            BasicLayer* getPrev () const;

				Matrix3D* calculateErrorCPU (Matrix3D* delta);

				Matrix3D* calculateErrorGPU(Matrix3D* delta);

				void updateWeightsGPU (Matrix3D* delta, double learningRate);

				void updateWeightsCPU(Matrix3D* delta, double learningRate);

            void toFile (std::ofstream* outputFile);
            
            static BasicLayer* loadFromFile (std::ifstream* inputFile, BasicLayer* prev = nullptr);
      };
		
   }
}
#endif