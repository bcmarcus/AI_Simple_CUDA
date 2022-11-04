#ifndef POOL_WEIGHT_HPP
#define POOL_WEIGHT_HPP

#include <iostream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "../weights/WeightBase.cuh"

using namespace coreutils::classes::matrixes;

namespace artificialIntelligence {
   namespace classes {

      // linked list holding all of the weights with each each weight
		// being represented by the input matrix and the output matrix.
		// first outputSize weights correspond to the first input etc.
      class PoolWeight : public WeightBase {
         private:
            Matrix3D* derivative;
            
				int poolLength;
				int poolWidth;
				int poolHeight;
            int size;
			public: 
				// -- CONSTRUCTOR DESTRUCTOR COPY -- //

				// default constructor
            PoolWeight ();

				// constructor
				// If randomize is 1, randomize using xavier Randomization. 
				// if randomize is 2, randomize between -0.5 and 0.5
				// else, dont randomize
            PoolWeight (int poolLength, int poolWidth, int poolHeight);

				// destructor
            ~PoolWeight ();

            // -- GET METHODS -- //

				// gets size
            long long getSize ();


				// gets the matrix in the linked list
				Matrix3D* getDerivativeMatrix () const;

				void setAll (double x);

            // -- PRINT METHODS -- // 

				// prints all of the weights
				void print ();

				void printPool ();

				long long paramCount ();
      };
   }
}


#endif