#include <Magick++.h> 

#include <artificialIntelligence/functions/images/generateInput.hpp>
#include <coreutils/classes/matrixes/Matrix3D.cuh>

using namespace Magick;
using namespace coreutils::classes::matrixes;

namespace artificialIntelligence {
   namespace functions {
      namespace images {
         namespace generate {
            void inputMatrixNormalized (std::string filepath, Matrix3D** inputMatrixes, int index, std::string type) {
               Image image(filepath);
               Geometry dimensions = image.size();
         
               if (type == "BW") {
                  inputMatrixes[index] = new Matrix3D(1, (int) dimensions.width(), (int) dimensions.height());
               }
               else if (type == "RGB") {
                  inputMatrixes[index] = new Matrix3D(3, (int) dimensions.width(), (int) dimensions.height());
               }
               else {
                  inputMatrixes[index] = new Matrix3D(4, (int) dimensions.width(), (int) dimensions.height());
               }

               for (size_t i = 0; i < dimensions.width(); i++) {
                  for (size_t j = 0; j < dimensions.height(); j++) {
                     Color c = image.pixelColor( i, j );
                     float r = c.quantumRed() / QuantumRange;
                     float g = c.quantumGreen() / QuantumRange;
                     float b = c.quantumBlue() / QuantumRange;
                     float a = c.quantumAlpha() / QuantumRange;
                     inputMatrixes[index]->insert(r, 0, i, j);
                     if (type != "BW") {
                        inputMatrixes[index]->insert(g, 1, i, j);
                        inputMatrixes[index]->insert(b, 2, i, j);
                        if (type != "RGB") {
                           inputMatrixes[index]->insert(a, 3, i, j);
                        }
                     }
                  }
               }
            }
         }
      }
   }
}