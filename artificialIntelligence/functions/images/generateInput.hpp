#ifndef IMAGES_GENERATE_HPP
#define IMAGES_GENERATE_HPP

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <string>

using namespace coreutils::classes::matrixes;

namespace artificialIntelligence {
   namespace functions {
      namespace images {
         namespace generate {
            void inputMatrixNormalized (std::string filepath,  Matrix3D** inputMatrixes, int index, std::string type = "RGBA");
         }
      }
   }
}

#endif