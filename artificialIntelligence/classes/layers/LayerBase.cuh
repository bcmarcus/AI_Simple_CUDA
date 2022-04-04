#ifndef LAYER_TEMPLATE
#define LAYER_TEMPLATE

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "../weights/WeightBase.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

namespace artificialIntelligence {
   namespace classes {
		class LayerBase {
         protected:
				// layer values
            Matrix3D* layerMatrix;

				// next biases
            Matrix3D** biasMatrixes;

				// next weights
            WeightBase** weights;

            // next layers
            LayerBase** next;

            // previous layers
            LayerBase** prev;

			public:
				LayerBase* getNext (int index = 0) const {
					return this->next[index];
				}

				LayerBase* getPrev (int index = 0) const {
					return this->prev[index];
				}

				LayerBase* getLast (int index = 0) {
					if ((this->next[index]) == nullptr) {
						return this;
					}
					return this->next[index]->getLast();
				}

				Matrix3D* getLayer () const {
					return this->layerMatrix;
				}

				Matrix3D* getBias (int index = 0) const {
					return this->biasMatrixes[index];
				}

				WeightBase* getWeights (int index = 0) const {
					return this->weights[index];
				}

				// sets the current layer matrix
            void setLayer (Matrix3D* layerMatrix) {
					if (this->layerMatrix == nullptr) {
						this->layerMatrix = new Matrix3D (layerMatrix->getLength(), layerMatrix->getWidth(), layerMatrix->getHeight());
					}
					this->layerMatrix->setMatrix(layerMatrix);
				}

				
		};
	}
}

#endif