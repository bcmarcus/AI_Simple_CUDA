#ifndef BASIC_WEIGHT_HPP
#define BASIC_WEIGHT_HPP

#include <iostream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "../weights/WeightBase.cuh"

using namespace coreutils::classes::matrixes;

namespace artificialIntelligence {
   namespace classes {

      // linked list holding all of the weights with each each weight
		// being represented by the input matrix and the output matrix.
		// first outputSize weights correspond to the first input etc.
      class BasicWeight : public WeightBase {
         private:         
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
            
			public: 
				// -- CONSTRUCTOR DESTRUCTOR COPY -- //

				// default constructor
            BasicWeight ();

				// constructor
				// If randomize is 1, randomize using xavier Randomization. 
				// if randomize is 2, randomize between -0.5 and 0.5
				// else, dont randomize
            BasicWeight (int fl, int fw, int fh, int sl, int sw, int sh, int randomize = 1);

				// destructor
            ~BasicWeight ();

				// copy constructor
				BasicWeight(const BasicWeight &w1);


            // -- GET METHODS -- //

				// gets size
            long long getSize ();

				// gets output size
            long long getOutputSize ();

				// gets weight data at a specific place
            float* getData (int fl, int fw, int fh, int sl, int sw, int sh);

				// gets the index of the weight assuming the weight is contiguous memory
				long long getIndex (int fl, int fw, int fh, int sl, int sw, int sh);

				// gets the matrix in the linked list
				Matrix3D* getWeightMatrix (int weightMatrixIndex = 0) const;


            // -- SET METHODS -- //

				// sets every value in the weight matrix to x
            void setAll (double x);

				// sets the value at a specific point to data
            void insertData (float data, int fl, int fw, int fh, int sl, int sw, int sh);


            // -- GENERATE METHODS -- //

            // builds a new weight matrix generating it based on given parameters
				// If randomize is 1, randomize using xavier Randomization. 
				// if randomize is 2, randomize between -0.5 and 0.5
				// else, dont randomize
				void build(int outputSize, int size, int toAdd, int length, int width, int height, int outputLength, int outputWidth, int outputHeight, int randomize);

            // builds a new weight with the input matrix as the weight matrix
				void build(int outputSize, int size, int length, int width, int height, int outputLength, int outputWidth, int outputHeight, const Matrix3D& inputMatrix);
				

            // -- PRINT METHODS -- // 

				// prints all of the weights
				void print ();

				long long paramCount ();
      };
   }
}


#endif