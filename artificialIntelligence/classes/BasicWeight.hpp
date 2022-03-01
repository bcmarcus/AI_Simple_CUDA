#ifndef BASIC_WEIGHT_HPP
#define BASIC_WEIGHT_HPP

#include <iostream>

#include <coreutils/classes/matrixes/Matrix3D.cpp>

using namespace coreutils::classes::matrixes;

namespace artificialIntelligence {
   namespace classes {

      
      class BasicWeight {
         public:
            BasicWeight (Matrix3D* weights);
            
            // generates weights going from of size fl, fw, fh to layer of size sl, sw, sh
            // names correspond to first, second, length, width, height
            BasicWeight (int fl, int fw, int fh, int sl, int sw, int sh);

            // generates a null weight            
            BasicWeight ();

            ~BasicWeight ();

            // prints the weight out in a natural format
            void print ();

            // 
            BasicWeight* add (int length, int width, int height, Matrix3D* weights = nullptr);

            BasicWeight* addNew (int length, int width, int height);

            Matrix3D* getWeightMatrix (int length, int width, int height);

            Matrix3D* getWeightMatrix ();

            float* getData (int fl, int fw, int fh, int sl, int sw, int sh);

            void insert (float data, int fl, int fw, int fh, int sl, int sw, int sh);
         private:

            // Cube where start is at the front top left corner.

            Matrix3D* weights;

            // corresponds to length
            BasicWeight* right;

            // corresponds to width;
            BasicWeight* back;

            // corresponds to height
            BasicWeight* down;

            // prints the matrixes in a reasonable manner with a count of them
            int print (int length, int width, int height);

            // generates nodes along the entire rectangular prism of side lengths from parameters
            int generate (int fl, int fw, int fh, int sl, int sw, int sh);
      };
   }
}


#endif