#ifndef ACTIVATION_FUNCTIONS_CUH
#define ACTIVATION_FUNCTIONS_CUH

#include <coreutils/classes/matrixes/Matrix3D.cuh>

using namespace coreutils::classes::matrixes;

namespace artificialIntelligence {
   namespace functions {
      namespace activation {
         // sigmoid function
         double sigmoid (double x);

         // derivative of the sigmoid function
         double dSigmoid (double x);

         // sigmoid function for entire matrix
         
         Matrix3D* sigmoid (Matrix3D* m3d, bool returnNew);

         // derivative of the sigmoid function for entire matrix
         
         Matrix3D* dSigmoid (Matrix3D* m3d);

         //sigmoid function
         double tanh (double x);

         // derivative of the tanH function
         double dTanh (double x);

         // tanh function for entire matrix
         
         Matrix3D* tanh (Matrix3D* m3d);

         // derivative of the tanh function for entire matrix
         
         Matrix3D* dTanh (Matrix3D* m3d);

			__device__ double device_sigmoid(double x);

			__device__ double device_dSigmoid(double x);

			__device__ double device_tanh(double x);

			__device__ double device_dTanh(double x);
      }
   }
}

#endif