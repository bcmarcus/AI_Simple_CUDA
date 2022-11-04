#include <cmath>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "activationFunctions.cuh"

using namespace coreutils::classes::matrixes;

namespace artificialIntelligence {
   namespace functions {
      namespace activation {

         double activate (ActivationType type, double x) {
            switch (type) {
               case ActivationType::Sigmoid:
                  return sigmoid (x);
               case ActivationType::Tanh:
                  return tanh (x);
               // case ActivationType::Relu:
               //    return relu (x);
               case ActivationType::LeakyRelu:
                  return leakyRelu (x);
               default:
                  return sigmoid (x);
            };
         }

         Matrix3D* activate (ActivationType type, Matrix3D* m3d) {
            switch (type) {
               case ActivationType::Sigmoid:
                  return sigmoid (m3d);
               case ActivationType::Tanh:
                  return tanh (m3d);
               // case ActivationType::Relu:
               //    return relu (m3d);
               case ActivationType::LeakyRelu:
                  return leakyRelu (m3d);
               default:
                  return sigmoid (m3d);
            };
         }

         double dActivate (ActivationType type, double x) {
            switch (type) {
               case ActivationType::Sigmoid:
                  return dSigmoid (x);
               case ActivationType::Tanh:
                  return dTanh (x);
               // case ActivationType::Relu:
               //    return relu (x);
               case ActivationType::LeakyRelu:
                  return dLeakyRelu (x);
               default:
                  return dSigmoid (x);
            };
         }

         Matrix3D* dActivate (ActivationType type, Matrix3D* m3d) {
            switch (type) {
               case ActivationType::Sigmoid:
                  return dSigmoid (m3d);
               case ActivationType::Tanh:
                  return dTanh (m3d);
               // case ActivationType::Relu:
               //    return relu (m3d);
               case ActivationType::LeakyRelu:
                  return dLeakyRelu (m3d);
               default:
                  return dSigmoid (m3d);
            };
         }

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
         
         
         Matrix3D* sigmoid(Matrix3D* m3d) 
         { 
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

         double leakyRelu(double x) {
            if (x > 0) {
               return x;
            }
            return x * LEAK_RATE;
			}

         double dLeakyRelu(double x) {
				if (x > 0) {
               return 1;
            }
				return LEAK_RATE;
			}

         Matrix3D* leakyRelu(Matrix3D* m3d) {
            Matrix3D* returnMatrix = new Matrix3D (m3d->getLength(), m3d->getWidth(), m3d->getHeight());
            for (int l = 0; l < m3d->getLength(); l++) {
               for (int w = 0; w < m3d->getWidth(); w++) {
                  for (int h = 0; h < m3d->getHeight(); h++) {
                     returnMatrix->insert(leakyRelu (*m3d->getData(l, w, h)), l, w, h);
                  }
               }
            }
            return returnMatrix;
			}

         Matrix3D* dLeakyRelu(Matrix3D* m3d) {
            Matrix3D* returnMatrix = new Matrix3D (m3d->getLength(), m3d->getWidth(), m3d->getHeight());
            for (int l = 0; l < m3d->getLength(); l++) {
               for (int w = 0; w < m3d->getWidth(); w++) {
                  for (int h = 0; h < m3d->getHeight(); h++) {
                     returnMatrix->insert(dLeakyRelu (*m3d->getData(l, w, h)), l, w, h);
                  }
               }
            }
            return returnMatrix;
			}

         __device__ double device_activate (ActivationType type, double x) {
            switch (type) {
               case ActivationType::Sigmoid:
                  return device_sigmoid (x);
               case ActivationType::Tanh:
                  return device_tanh (x);
               // case ActivationType::Relu:
               //    return relu (x);
               case ActivationType::LeakyRelu:
                  return device_leakyRelu (x);
               default:
                  return device_sigmoid (x);
            };
         }

         __device__ double device_dActivate (ActivationType type, double x) {
            switch (type) {
               case ActivationType::Sigmoid:
                  return device_dSigmoid (x);
               case ActivationType::Tanh:
                  return device_dTanh (x);
               // case ActivationType::Relu:
               //    return relu (x);
               case ActivationType::LeakyRelu:
                  return device_dLeakyRelu (x);
               default:
                  return device_dSigmoid (x);
            };
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

         __device__ double device_leakyRelu(double x) {
            if (x > 0) {
               return x;
            }
            return x * LEAK_RATE;
			}

         __device__ double device_dLeakyRelu(double x) {
				if (x > 0) {
               return 1;
            }
				return LEAK_RATE;
			}
      }
   }
}
