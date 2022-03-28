#ifndef BASIC_LAYER_HPP
#define BASIC_LAYER_HPP

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include "BasicWeight.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

namespace artificialIntelligence {
   namespace classes {

      // -- CUDA FUNCTIONS -- //

      // updates this layer using GPU compute
		__global__ void calculateAndUpdateLayerGPU (float* nodeValues, float* weights, float* output, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId);

      // calculates the error in the layer using CPU compute
		__global__ void calculateError (float* weights, float* delta, float* error, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId);

      // updates weights in the layer using CPU compute
		__global__ void updateWeights (float* weights, float* delta, float* input, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId, double learningRate);

      class BasicLayer{
         private:
            // layer values
            Matrix3D* layerMatrix;
            Matrix3D* biasMatrix;
            BasicWeight* weights;

            // next layer
            BasicLayer* next;

            // previous layer
            BasicLayer* prev;

         public:

            // -- CONSTRUCTOR DESTRUCTOR COPY -- //
            
            // constructors
            BasicLayer (Matrix3D* layerMatrix, Matrix3D* biasMatrix = nullptr, BasicWeight* weights = nullptr);
            BasicLayer (int length, int width, int height);
            BasicLayer ();

            // destructor
            ~BasicLayer ();
            
            // copy constructor
				BasicLayer (const BasicLayer& b, bool copyAll = false);

            
            // -- GET METHODS -- //

            // gets the next layer in the model
            BasicLayer* getNext () const;

            // gets the previous layer in the model
            BasicLayer* getPrev () const;

            // gets the last layer in the model
            BasicLayer* getLast ();

            // gets the current layer matrix
            Matrix3D* getLayer () const;

            // gets the current bias matrix
            Matrix3D* getBias () const;

            // gets the current weights
				BasicWeight* getWeights () const;


            // -- SET METHODS -- //

            // sets the next layer in the model
            void setNext (BasicLayer* next);

            // sets the previous layer in the model
            void setPrev (BasicLayer* prev);

            // sets the current layer matrix
            void setLayer (Matrix3D* layerMatrix);

            // sets the current bias matrix
				void setBias (Matrix3D* biasMatrix);

            // sets the current weights
				void setWeights (BasicWeight* weight);
            

            // -- GENERATE METHODS -- // 

            // adds a layer at the end of the model recursively
            BasicLayer* add (BasicLayer* layer);

            // creates and adds a layer at the end of the model recursively
            BasicLayer* add (Matrix3D* layer, Matrix3D* biasMatrix = nullptr, BasicWeight* weights = nullptr);

            // creates a new weight based on the two given layers
				BasicWeight* newWeight (BasicLayer* firstLayer, BasicLayer* secondLayer);


            // -- LAYER UPDATE METHODS -- //

            // updates all layers in the model using CPU compute
            void calculateAndUpdateAllCPU ();

            // updates this layer using CPU compute
            void calculateAndUpdateLayerCPU ();

            // updates all layers in the model using GPU compute revised
				void calculateAndUpdateAllGPUV2 ();

            // updates this layer using GPU compute
			 	void calculateAndUpdateLayerGPU();

            // calculates the error in the layer using CPU compute
				Matrix3D* calculateErrorCPU (Matrix3D* delta);

            // calculates the error in the layer using GPU compute
				Matrix3D* calculateErrorGPU (Matrix3D* delta);

            // updates weights in the layer using CPU compute
				void updateWeightsCPU(Matrix3D* delta, double learningRate);

            // updates weights in the layer using GPU compute
				void updateWeightsGPU (Matrix3D* delta, double learningRate);


            // -- PRINT METHODS -- //
   
            // prints the layer and all layers below it
            int print (bool printBias = false, bool printWeights = false, int depth = 1);


            // -- LOAD AND UNLOAD FILE METHODS -- //

            // loads a model into a file using the format of 
            // layer length, layer width, layer height
            // bias length, bias width, bias height
            // <the values for the bias, all comma seperated>
            // layer length, layer width, layer height, bias length, bias width, bias height
            // <the values for the weights, with each float16 represented by 4 bytes of data> 
            void toFile (std::ofstream* outputFile);
            
            // loads a model from a file using the format described above
            static BasicLayer* loadFromFile (std::ifstream* inputFile, BasicLayer* prev = nullptr);
      };
		
   }
}
#endif