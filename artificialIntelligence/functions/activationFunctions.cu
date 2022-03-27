#include <cmath>
#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include "activationFunctions.cuh"

using namespace coreutils::classes::matrixes;

namespace artificialIntelligence {
   namespace functions {
      namespace activation {

         double sigmoid(double x) 
         { 
				if (x > 20) {
					return 1;
				}
				if (x < 0.000001 && x > -0.000001) {
					return 0.5;
				}
            return 1 / (1 + exp(-x)); 
         }

         double dSigmoid(double x) 
         {
				if (x > 20) {
					return 0;
				}
				if (x < 0.000001 && x > -0.000001) {
					return 0.25;
				}
            return sigmoid(x) * (1 - sigmoid(x));
         }
         
         
         Matrix3D* sigmoid(Matrix3D* m3d, bool returnNew) 
         { 
				if (returnNew) {
					Matrix3D* returnMatrix = new Matrix3D (m3d->getLength(), m3d->getWidth(), m3d->getHeight());
					for (int l = 0; l < m3d->getLength(); l++) {
						for (int w = 0; w < m3d->getWidth(); w++) {
							for (int h = 0; h < m3d->getHeight(); h++) {
								returnMatrix->insert(sigmoid (*m3d->getData(l, w, h)), l, w, h);
							}
						}
					}
					return returnMatrix;
				} 
				for (int l = 0; l < m3d->getLength(); l++) {
					for (int w = 0; w < m3d->getWidth(); w++) {
						for (int h = 0; h < m3d->getHeight(); h++) {
							m3d->insert(sigmoid (*m3d->getData(l, w, h)), l, w, h);
						}
					}
				}
				return m3d;
         }

         
         Matrix3D* dSigmoid(Matrix3D* m3d) 
         { 
            Matrix3D* returnMatrix = new Matrix3D (m3d->getLength(), m3d->getWidth(), m3d->getHeight());
            for (int l = 0; l < m3d->getLength(); l++) {
               for (int w = 0; w < m3d->getWidth(); w++) {
                  for (int h = 0; h < m3d->getHeight(); h++) {
                     returnMatrix->insert(dSigmoid (*m3d->getData(l, w, h)), l, w, h);
                  }
               }
            }
            return returnMatrix;
         }

         double tanh (double x) {
            return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
         }

         double dTanh (double x) {
            return pow(tanh(x), 2);
         }

         
         Matrix3D* tanh(Matrix3D* m3d) 
         { 
            Matrix3D* returnMatrix = new Matrix3D (m3d->getLength(), m3d->getWidth(), m3d->getHeight());
            for (int l = 0; l < m3d->getLength(); l++) {
               for (int w = 0; w < m3d->getWidth(); w++) {
                  for (int h = 0; h < m3d->getHeight(); h++) {
                     returnMatrix->insert(tanh (*m3d->getData(l, w, h)), l, w, h);
                  }
               }
            }
            return returnMatrix;
         }

         
         Matrix3D* dTanh(Matrix3D* m3d) 
         { 
            Matrix3D* returnMatrix = new Matrix3D (m3d->getLength(), m3d->getWidth(), m3d->getHeight());
            for (int l = 0; l < m3d->getLength(); l++) {
               for (int w = 0; w < m3d->getWidth(); w++) {
                  for (int h = 0; h < m3d->getHeight(); h++) {
                     returnMatrix->insert(dTanh (*m3d->getData(l, w, h)), l, w, h);
                  }
               }
            }
            return returnMatrix;
         }


			__device__ double device_sigmoid(double x) {
				if (x > 20) {
					return 1;
				}
				return 1 / (1 + expf(-x));
			}

			__device__ double device_dSigmoid(double x) {
				return device_sigmoid(x) * (1 - device_sigmoid(x));
			}

			__device__ double device_tanh(double x) {
				return tanhf(x);
			}

			__device__ double device_dTanh(double x) {
				double y = tanhf(x);
				return y * y;
			}
      }
   }
}
