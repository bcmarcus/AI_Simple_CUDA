#ifndef GENERATIONAL_AI_BASIC_CUH
#define GENERATIONAL_AI_BASIC_CUH

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <artificialIntelligence/classes/BasicLayerList.hpp>

namespace artificialIntelligence {
   namespace basicLearningTypes {
      namespace generationalAIBasic {
			
			__global__ void sumMultiplyOfMatrixesDevice (float* device_weights, float* device_delta, float* output);

		}
	}
}

#endif