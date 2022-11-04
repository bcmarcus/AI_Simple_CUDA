#ifndef WEIGHT_TEMPLATE
#define WEIGHT_TEMPLATE

#include <iostream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

using namespace coreutils::classes::matrixes;

#define WEIGHT_MAX_SIZE (1024*1024*256)
// #define WEIGHT_MAX_SIZE (1024*1024*8)

namespace artificialIntelligence {
   namespace classes {
      class WeightBase {
         public:
            virtual void setAll (double x) = 0;

			   virtual void print () = 0;

            virtual long long paramCount () = 0;
		};
	}
}

#endif