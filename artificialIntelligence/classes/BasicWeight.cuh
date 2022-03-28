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
            
            BasicWeight ();
            BasicWeight (int fl, int fw, int fh, int sl, int sw, int sh, bool randomize = true);

            ~BasicWeight ();

				BasicWeight(const BasicWeight &w1);

            // -- GET METHODS -- //
            long long getSize ();

            float* getData (int fl, int fw, int fh, int sl, int sw, int sh);

				long long getIndex (int fl, int fw, int fh, int sl, int sw, int sh);

				Matrix3D* getWeightMatrix (int index) const;

            // -- SET METHODS -- //
            void setAll (double x);

            void insertData (float data, int fl, int fw, int fh, int sl, int sw, int sh);


            // -- GENERATE METHODS -- //

            // builds a new weight matrix generating it based on given parameters
				void build(int outputSize, int size, int toAdd, int length, int width, int height, int outputLength, int outputWidth, int outputHeight, bool randomize);

            // builds a new weight with the input matrix as the weight matrix
				void build(int outputSize, int size, int length, int width, int height, int outputLength, int outputWidth, int outputHeight, const Matrix3D& inputMatrix);
				
            // -- PRINT METHODS -- // 
				int print ();
            int print (int length, int width, int height);
      };
   }
}


#endif