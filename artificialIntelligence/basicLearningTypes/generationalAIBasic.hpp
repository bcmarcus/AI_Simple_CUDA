#ifndef GENERATIONAL_AI_BASIC_HPP
#define GENERATIONAL_AI_BASIC_HPP

#include <coreutils/classes/matrixes/Matrix3D.cpp>
#include <artificialIntelligence/classes/BasicLayerList.hpp>

namespace artificialIntelligence {
   namespace basicLearningTypes {
      namespace generationalAIBasic {

			float sumMultiplyOfMatrixes(Matrix3D* first, Matrix3D* second);

         void multiplyMatrixes(Matrix3D* first, Matrix3D* second, Matrix3D* output);
            
         void run (
				BasicLayerList* list, 
				int epochs, 
				double learningRate, 
				Matrix3D** inputDataMatrixes, 
				Matrix3D** outputDataMatrixes, 
				int inputCount, 
				bool calculateError = false, 
				bool print = false
			);
		}
	}
}

#endif