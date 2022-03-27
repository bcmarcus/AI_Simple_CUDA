#ifndef BASIC_WEIGHT_HPP
#define BASIC_WEIGHT_HPP

#include <iostream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

using namespace coreutils::classes::matrixes;


#define BASIC_WEIGHT_MAX_SIZE (1024*1024*256)

// #define BASIC_WEIGHT_MAX_SIZE (1024*1024)
namespace artificialIntelligence {
   namespace classes {

      
      class BasicWeight {
         public:
            
            // generates weights going from of size fl, fw, fh to layer of size sl, sw, sh
            // names correspond to first, second, length, width, height
            BasicWeight (int fl, int fw, int fh, int sl, int sw, int sh, bool randomize = true);

				void build(int outputSize, int size, int toAdd, int length, int width, int height, int outputLength, int outputWidth, int outputHeight, bool randomize);

				void build(int outputSize, int size, int length, int width, int height, int outputLength, int outputWidth, int outputHeight, const Matrix3D& inputMatrix);

				// BasicWeight (int size, int fl, int fh, int fw, int sl, int sw, int sh);

				BasicWeight ();

            ~BasicWeight ();

				BasicWeight(const BasicWeight &w1);

				Matrix3D* getWeightMatrix (int index) const;

				long long getIndex (int fl, int fw, int fh, int sl, int sw, int sh);

				long long getSize ();

            float* getData (int fl, int fw, int fh, int sl, int sw, int sh);

            void insertData (float data, int fl, int fw, int fh, int sl, int sw, int sh);

				void setAll (double x);
            // Cube where start is at the front top left corner.

				// matrix where each matrix is defined by its length, width, and height, and anything after that is the next matrix
            Matrix3D* weights;
            BasicWeight* next;
				
				long long size;
				int outputSize;
				int length;
				int width;
				int height;
				int outputLength;
				int outputWidth;
				int outputHeight;
				
				int print ();
            // prints the matrixes in a reasonable manner with a count of them
            int print (int length, int width, int height);
      };
   }
}


#endif