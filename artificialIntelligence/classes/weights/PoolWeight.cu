#include <iostream>
#include <string>
#include <cmath>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "../weights/PoolWeight.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

// default constructor
PoolWeight::PoolWeight () {
   this->derivative = nullptr;
   this->poolLength = 0;
   this->poolWidth = 0;
   this->poolHeight = 0;
   this->size = 0;
}

// constructor
// If randomize is 1, randomize using xavier Randomization. 
// if randomize is 2, randomize between -0.5 and 0.5
// else, dont randomize
PoolWeight::PoolWeight (int poolLength, int poolWidth, int poolHeight) {
   this->derivative = new Matrix3D (poolLength, poolWidth, poolHeight);
   this->poolLength = poolLength;
   this->poolWidth = poolWidth;
   this->poolHeight = poolHeight;
   this->size = derivative->getSize ();
}

// destructor
PoolWeight::~PoolWeight () {
   delete this->derivative;
}

// -- GET METHODS -- //

// gets size
long long PoolWeight::getSize () {
   return this->size;
}


// gets the matrix in the linked list
Matrix3D* PoolWeight::getDerivativeMatrix () const {
   return this->derivative;
}

void PoolWeight::setAll (double x) {
	this->derivative->setAll(x);
}

// -- PRINT METHODS -- // 

// prints all of the weights
void PoolWeight::print () {
   this->derivative->printMatrix();
}

void PoolWeight::printPool () {
   this->derivative->printMatrixSize();
}

long long PoolWeight::paramCount () {
	return 0;
}