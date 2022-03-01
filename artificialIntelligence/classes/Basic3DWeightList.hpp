#ifndef BASIC_3D_WEIGHT_LIST_HPP
#define BASIC_3D_WEIGHT_LIST_HPP

#include <coreutils/classes/matrixes/Matrix3D.cpp>
#include <artificialIntelligence/classes/BasicWeight.hpp>

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

namespace artificialIntelligence {
   namespace classes {
      
      
      class Basic3DWeightList {
         public:
            Basic3DWeightList (int fl, int fw, int fh, int sl, int sw, int sh);

            Basic3DWeightList ();

            void print ();

            Matrix3D* getWeightMatrix (int length, int width, int height);

            void insert (int length, int width, int height, Matrix3D weights);

         private:
            BasicWeight* root;
      };
   }
}

#endif