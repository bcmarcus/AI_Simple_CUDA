#ifndef Pool_LAYER_HPP
#define Pool_LAYER_HPP

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "LayerBase.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

namespace artificialIntelligence {
   namespace classes {

      // -- CUDA FUNCTIONS -- //

      // updates this layer using GPU compute
		__global__ void calculateAndUpdateLayerGPUPool (float* nodeValues, float* weights, float* output, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId);

      // calculates the error in the layer using CPU compute
		__global__ void calculateErrorPool (float* weights, float* delta, float* error, int inputSize, int outputSize, int numPerThread, long long maxWeightIndex, long long helperIndex, long long startingWeight, int startingOutputId);

      class PoolLayer : public LayerBase {
         protected:
            int poolLength;
            int poolWidth;
            int poolHeight;
         public:

            // -- CONSTRUCTOR DESTRUCTOR COPY -- //
            
            // default coonstructor
				PoolLayer (Matrix3D* layerMatrix, int poolLength, int poolWidth, int poolHeight, int layerMatrixCount, ActivationType activationType);

				// generate layer constructor
            PoolLayer (int length, int width, int height, int poolLength, int poolWidth, int poolHeight, int layerMatrixCount = 1, ActivationType activationType = ActivationType::Sigmoid);

				// initialize with values constructor
            PoolLayer ();

            // destructor
            ~PoolLayer ();
            
            // copy constructor
				PoolLayer (const PoolLayer& b, bool copyAll = false);

            
            // -- GET METHODS -- //

            int getPoolLength () const;

            int getPoolWidth () const;

            int getPoolHeight () const;

            // gets the next layer in the model
            PoolLayer* getNext (int index = 0) const;

            // gets the previous layer in the model
            // PoolLayer* getPrev () const;

            // gets the last layer in the model
            // PoolLayer* getLast ();

            // gets the current layer matrix
            // Matrix3D* getLayer () const;

            // -- SET METHODS -- //

            // sets the next layer in the model
            // void setNext (PoolLayer* next);

            // sets the previous layer in the model
            // void setPrev (PoolLayer* prev);

            // sets the current layer matrix
            // void setLayer (Matrix3D* layerMatrix);
            

            // -- GENERATE METHODS -- // 

            // adds a layer at the end of the model recursively
            // PoolLayer* add (LayerBase* layer);

            // creates and adds a layer at the end of the model recursively
            PoolLayer* add (Matrix3D* layerMatrix, int poolLength, int poolWidth, int poolHeight, int layerMatrixCount, ActivationType activationType);

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