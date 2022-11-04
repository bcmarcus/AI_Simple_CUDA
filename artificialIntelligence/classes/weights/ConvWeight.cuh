#ifndef Conv_WEIGHT_HPP
#define Conv_WEIGHT_HPP

#include <iostream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "../weights/WeightBase.cuh"

using namespace coreutils::classes::matrixes;

namespace artificialIntelligence {
   namespace classes {

      // linked list holding all of the weights with each each weight
		// being represented by the input matrix and the output matrix.
		// first outputSize weights correspond to the first input etc.
      class ConvWeight : public WeightBase {
         private:         
            Matrix3D** weights;
				long long size;

				int convLength;
				int convWidth;
				int convHeight;
            int features;
            int stride;
            
			public: 
				// -- CONSTRUCTOR DESTRUCTOR COPY -- //

				// default constructor
            ConvWeight ();

				// constructor
				// If randomize is 1, randomize using xavier Randomization. 
				// if randomize is 2, randomize between -0.5 and 0.5
				// else, dont randomize
            ConvWeight (int convLength, int convWidth, int convHeight, int features, int stride = 1, int randomize = 1);

				// destructor
            ~ConvWeight ();

            // -- GET METHODS -- //

				// gets size
            long long getSize ();

				// gets weight data at a specific place
            float* getData (int convLengthIndex, int convWidthIndex, int convHeightIndex, int featureIndex);

				// gets the index of the weight assuming the weight is contiguous memory
				long long getIndex (int convLengthIndex, int convWidthIndex, int convHeightIndex, int featureIndex);

				// gets the matrix in the linked list
				Matrix3D* getWeightMatrix (int feature = 0) const;


            // -- SET METHODS -- //

				// sets every value in the weight matrix to x
            void setAll (double x);

				// sets the value at a specific point to data
            void insertData (float data, int convLengthIndex, int convWidthIndex, int convHeightIndex, int featureIndex);

            // -- PRINT METHODS -- // 

				// prints all of the weights
				void print ();

				void printConv ();

				long long paramCount ();
      };
   }
}


#endif