#include <iostream>
#include <string>
#include <cmath>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "../weights/ConvWeight.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;


// default constructor
ConvWeight::ConvWeight () {
	this->weights = nullptr;
	this->size = 0;

	this->convLength = 0;
	this->convWidth = 0;
	this->convHeight = 0;
	this->features = 0;
	this->stride = 0;
}

// constructor
// If randomize is 1, randomize using xavier Randomization. 
// if randomize is 2, randomize between -0.5 and 0.5
// else, dont randomize
ConvWeight::ConvWeight (int convLength, int convWidth, int convHeight, int features, int stride, int randomize) {
	
	// if the values are bad they break the program so error
	try {
      if (convLength < 1 || convWidth < 1 || convHeight < 1) {
         std::cout << " convLength :: " << convLength << " convWidth :: " << convWidth << " convHeight :: " << convHeight;
         throw std::invalid_argument("\nConvWeight generate values are invalid\n\n");
      }
   } catch(const std::invalid_argument& e) {
      std::cout << e.what();
      exit (0);
   };

	// initializes all member variables
   this->weights = new Matrix3D* [features];
	this->size = convLength * convWidth * convHeight * features;

	this->convLength = convLength;
	this->convWidth = convWidth;
	this->convHeight = convHeight;
	this->features = features;
	this->stride = stride;

	for (int i = 0; i < features; i++) {
		this->weights[i] = new Matrix3D(convLength, convWidth, convHeight);
		this->weights[i]->randomize ();
	}
}

// destructor
ConvWeight::~ConvWeight(){
   if (this->weights != nullptr) {
		for (int i = 0; i < features; i++) {
      	delete this->weights[i];
		}
   } else {
      return;
   }
}

// -- GET METHODS -- //

// gets size
long long ConvWeight::getSize(){
	return this->size;
}

// gets weight data at a specific place
float* ConvWeight::getData (int feature, int convLength, int convWidth, int convHeight) {
	return this->weights[feature]->getData(convLength, convWidth, convHeight);
}

// gets the matrix in the linked list
Matrix3D* ConvWeight::getWeightMatrix (int feature) const {
	return this->weights[feature];
}

// -- SET METHODS -- //

// sets every value in the weight matrix to x
void ConvWeight::setAll (double x) {
	for (int i = 0 ; i < this->features; i++) {
		this->weights[i]->setAll(x);
	}
}

// sets the value at a specific point to data
void ConvWeight::insertData (float data, int feature, int convLength, int convWidth, int convHeight) {
	this->weights[feature]->insert(data, convLength, convWidth, convHeight);
}

// -- PRINT METHODS -- // 

// prints all of the weights
void ConvWeight::print () {
	for (int i = 0; i < this->features; i++) {
		this->weights[i]->printMatrix();
	}
}

void ConvWeight::printConv () {
	std::cout << "Conv: " << this->convLength << "x" << this->convWidth << "x" << this->convHeight << " || Features: " << this->features << '\n';
}

long long ConvWeight::paramCount () {
	int count = 0;
	for (int i = 0; i < this->features; i++) {
		count += this->weights[i]->getSize() / sizeof(float);
	}
	return count;
}