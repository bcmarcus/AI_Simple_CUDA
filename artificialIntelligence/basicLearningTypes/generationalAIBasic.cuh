#ifndef GENERATIONAL_AI_BASIC_CUH
#define GENERATIONAL_AI_BASIC_CUH

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <artificialIntelligence/classes/layerLists/LayerList.cuh>

namespace artificialIntelligence {
   namespace basicLearningTypes {
      namespace generationalAIBasic {

			// runs a generic neural network for a certain number of epochs
			// batch size must be a multiple of inputCount or values will be truncated at the end
         void runStochasticGradientDescent (
				LayerList* list, 
				int epochs,
				int GPU,
				double learningRate, 
				Matrix3D** inputDataMatrixes, 
				Matrix3D** outputDataMatrixes, 
				int inputCount,
				int batchSize = 0,
				bool calculateError = false, 
				bool print = false
			);
		}
	}
}

#endif