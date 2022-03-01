#ifndef SORTHELPERS
#define SORTHELPERS

#include <iostream>
#include <string>
#include <unistd.h>
#include <coreutils/functions/math/simpleMath.hpp>

using namespace coreutils::functions;

namespace coreutils 
{
   namespace functions
   {
      namespace sort {
         template <typename T>
         void swap (T* first, T* second){
            T temp = *first;
            *first = *second;
            *second = temp;
         }

         template <typename T>
         void shuffle (T* arr, int size) {
            for (int i = 0; i < size; i++) {
               coreutils::functions::sort::swap (&arr[i], &arr[(int) math::rand(0, size - 1)]);
            }
         }

         template <typename T>
         void reverse (T* arr, int size) {
            for (int i = 0; i < size / 2; i++) {
               coreutils::functions::sort::swap (&arr[i], &arr[size - i - 1]);
            }
         }
      }
   }
}

#endif