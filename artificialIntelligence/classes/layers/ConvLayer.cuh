#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "LayerBase.cuh"
#include "../weights/ConvWeight.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

namespace artificialIntelligence {
   namespace classes {

      // -- CUDA FUNCTIONS -- //

      // updates this layer using GPU compute
		__global__ void calculateAndUpdateLayerGPUConv (float* nodeValues, float* weights, float* output, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId);

      // calculates the error in the layer using CPU compute
		__global__ void calculateErrorConv (float* weights, float* delta, float* error, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId);

      // updates weights in the layer using CPU compute
		__global__ void updateWeightsConv (float* weights, float* delta, float* input, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId, double learningRate);

      class ConvLayer : public LayerBase {
         protected:
            int convLength;
            int convWidth;
            int convHeight;
            int features;
            int stride;
         public:

            // -- CONSTRUCTOR DESTRUCTOR COPY -- //
            
            // default coonstructor
				ConvLayer ();

				// generate layer constructor
            ConvLayer (int length, int width, int height, int convLength = 1, int convWidth = 3, int convHeight = 3, int features = 1, int stride = 1, int layerMatrixCount = 1, ActivationType activationType = ActivationType::Sigmoid);

            // generate layer constructor
            ConvLayer (Matrix3D* layerMatrix, int convLength, int convWidth, int convHeight, ActivationType activationType);

				// initialize with values constructor
            ConvLayer (Matrix3D* layerMatrix, Matrix3D* biasMatrix = nullptr, ConvWeight* weights = nullptr, ActivationType activationType = ActivationType::Sigmoid);

            // destructor
            ~ConvLayer ();
            
            // copy constructor
				// ConvLayer (const ConvLayer& b, bool copyAll = false);

            
            // -- GET METHODS -- //

            int getFeatureCount () const;

            // gets the next layer in the model
            // ConvLayer* getNext () const;

            // gets the previous layer in the model
            // ConvLayer* getPrev () const;

            // gets the last layer in the model
            // ConvLayer* getLast ();

            // gets the current layer matrix
            // Matrix3D* getLayer () const;

            // gets the current bias matrix
            // Matrix3D* getBias () const;

            // gets the current weights
				// ConvWeight* getWeights () const;


            // -- SET METHODS -- //

            // sets the next layer in the model
            // void setNext (ConvLayer* next);

            // sets the previous layer in the model
            // void setPrev (ConvLayer* prev);

            // sets the current layer matrix
            // void setLayerMatrix (Matrix3D* layerMatrix);

            // sets the current bias matrix
				// void setBias (Matrix3D* biasMatrix);

            // sets the current weights
				// void setWeights (WeightBase* weight);
            

            // -- GENERATE METHODS -- // 

            // adds a layer at the end of the model recursively
            // ConvLayer* add (LayerBase* layer);

            // creates and adds a layer at the end of the model recursively
            // ConvLayer* add (Matrix3D* layer, Matrix3D* biasMatrix, ConvWeight* weights, int index);

            // creates a new weight based on the two given layers
				WeightBase* newWeight (int index);

            // creates a new bias based on the two given layers
            Matrix3D* newBias (int index);


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
            void printDetails ();


            // -- LOAD AND UNLOAD FILE METHODS -- //

            // loads a model into a file using the format of 
            // layer length, layer width, layer height
            // bias length, bias width, bias height
            // <the values for the bias, all comma seperated>
            // layer length, layer width, layer height, bias length, bias width, bias height
            // <the values for the weights, with each float16 represented by 4 bytes of data> 
            void toFile (std::ofstream* outputFile);
            
            // loads a model from a file using the format described above
            static LayerBase* loadFromFile (std::ifstream* inputFile, LayerBase* prev = nullptr);
      };
		
   }
}
#endif