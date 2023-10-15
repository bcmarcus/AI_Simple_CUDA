#ifndef BIAS_TEMPLATE
#define BIAS_TEMPLATE

#include <iostream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

using namespace coreutils::classes::matrixes;

namespace artificialIntelligence {
   namespace classes {
      class BiasBase {
         protected:
            Matrix3D** matrixes;

         public:
            

            virtual void setAll (double x) = 0;

			   virtual void print () = 0;

            virtual long long paramCount () = 0;
		};
	}
}

#endif