#ifndef ACTIVATION_FUNCTIONS_CUH
#define ACTIVATION_FUNCTIONS_CUH

#include <coreutils/classes/matrixes/Matrix3D.cuh>

using namespace coreutils::classes::matrixes;

#define LEAK_RATE 0.01

namespace artificialIntelligence {
   namespace functions {
      namespace activation {
         enum ActivationType {
            Sigmoid,
            Tanh,
            Relu,
            LeakyRelu
         };

         double activate (ActivationType type, double x);

         Matrix3D* activate (ActivationType type, Matrix3D* m3d);

         double dActivate (ActivationType type, double x);

         Matrix3D* dActivate (ActivationType type, Matrix3D* m3d);

         // sigmoid function
         double sigmoid (double x);

         // derivative of the sigmoid function
         double dSigmoid (double x);

         // sigmoid function for entire matrix
         
         Matrix3D* sigmoid (Matrix3D* m3d);
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

         //sigmoid function
         double leakyRelu (double x);

         // derivative of the tanH function
         double dLeakyRelu (double x);

         // tanh function for entire matrix
         
         Matrix3D* leakyRelu (Matrix3D* m3d);

         // derivative of the tanh function for entire matrix
         
         Matrix3D* dLeakyRelu (Matrix3D* m3d);

         __device__ double device_activate(ActivationType type, double x);
         
         __device__ double device_dActivate(ActivationType type, double x);

			__device__ double device_sigmoid(double x);

			__device__ double device_dSigmoid(double x);

			__device__ double device_tanh(double x);

			__device__ double device_dTanh(double x);

         __device__ double device_leakyRelu(double x);

         __device__ double device_dLeakyRelu(double x);
      }
   }
}

#endif