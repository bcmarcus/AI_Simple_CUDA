#include <iostream>
#include <string>
#include <cmath>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "../weights/BasicWeight.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;


// default constructor
BasicWeight::BasicWeight () {
	this->size = 0;
	this->weights = nullptr;
	this->next = nullptr;
	this->length = 0;
	this->width = 0;
	this->height = 0;
}

// constructor
// If randomize is 1, randomize using xavier Randomization. 
// if randomize is 2, randomize between -0.5 and 0.5
// else, dont randomize
BasicWeight::BasicWeight (int fl, int fw, int fh, int sl, int sw, int sh, int randomize) {
	
	// if the values are bad they break the program so error
	try {
      if (fl < 1 || fw < 1 || fh < 1 || sl < 1 || sw < 1 || sh < 1) {
         std::cout << " \nfl :: " << fl << " fw :: " << fw << " fh :: " << fh 
                  << " sl :: " << sl << " sw :: " << sw << " sh :: " << sh;
         throw std::invalid_argument("\nBasicWeight generate values are invalid\n\n");
      }
   } catch(const std::invalid_argument& e) {
      std::cout << e.what();
      exit (0);
   };

	// initializes all member variables
   this->weights = nullptr;
   this->next = nullptr;
	this->outputSize = sl * sw * sh;
	this->size = fl * fw * fh * outputSize;
	this->length = fl;
	this->width = fw;
	this->height = fh;
	this->outputLength = sl;
	this->outputWidth = sw;
	this->outputHeight = sh;

	// amount to add to the weight linked list
	long long amountLeftToAdd = this->size;
	BasicWeight* current = this;
	int toAdd = WEIGHT_MAX_SIZE;

	// begin looping and adding the maximum amount to each weight matrix
	while (amountLeftToAdd > 0) {

		// make sure the proper number of weights are added
		if (amountLeftToAdd < WEIGHT_MAX_SIZE) {
			toAdd = amountLeftToAdd;
		}

		// build a new weight link
		current->build(outputSize, this->size, toAdd, length, width, height, outputLength, outputWidth, outputHeight, randomize);
		
		// update the amount left to add
		amountLeftToAdd -= toAdd;
		
		// if not completed, keep making more links
		if (amountLeftToAdd > 0) {
			current->next = new BasicWeight();
			current = current->next;
		} else {
			current->next = nullptr;
		}
	}
}

// destructor
BasicWeight::~BasicWeight(){
   if (this->weights != nullptr) {
		delete this->next;
      delete this->weights;
   } else {
      return;
   }
}

// copy constructor
BasicWeight::BasicWeight(const BasicWeight &w){

	// initializations
	BasicWeight* currentCopy = this;
	int currentInputIndex = 0;

	// loop through and get copy every single link
	while (w.getWeightMatrix(currentInputIndex) != nullptr) {
		currentCopy->build(w.outputSize, w.size, w.length, w.width, w.height, w.outputLength, w.outputWidth, w.outputHeight, *(w.getWeightMatrix(currentInputIndex)));
		currentCopy->next = new BasicWeight();
		currentInputIndex++;
		if (w.getWeightMatrix(currentInputIndex) != nullptr) {
			currentCopy = currentCopy->next;
		}
	}

	// delete extra weight that was made
	delete currentCopy->next;
	currentCopy->next = nullptr;
}


// -- GET METHODS -- //

// gets size
long long BasicWeight::getSize(){
	return this->size;
}

// gets output size
long long BasicWeight::getOutputSize(){
	return this->outputSize;
}

// gets weight data at a specific place
float* BasicWeight::getData (int fl, int fw, int fh, int sl, int sw, int sh) {

	// initializations
	BasicWeight* current = this;
	long long index = getIndex(fl, fw, fh, sl, sw, sh);

	// go to the proper layer
	while (index >= current->weights->getSize() / sizeof(float)) {
		index -= current->weights->getSize() / sizeof(float);
		// std::cout << "indexUpdated: " << index << '\n';
		current = current->next;
	}
	
	//
	return current->weights->getData(0, 0, index);
}

// gets the index of the weight assuming the weight is contiguous memory
long long BasicWeight::getIndex (int fl, int fw, int fh, int sl, int sw, int sh) {

	// makes sure the weight exists
	if (fl > this->length || fw > this->width || fh > this->height || sl > this->outputLength || sw > this->outputWidth || sh > this->outputHeight) {
		std::cout << "Index out of bounds getIndex in BasicWeights\n";
		exit(0);
	}

	// return the value that the weight exists at
	return (fl * this->width * this->height + fw * this->height + fh) * outputSize  + sl * this->outputWidth * this->outputHeight + sw * this->outputHeight + sh;
}

// gets the matrix in the linked list
Matrix3D* BasicWeight::getWeightMatrix (int weightMatrixIndex) const {

	// initialization
	const BasicWeight* current = this;

	// run through the links until the proper one is found
	while (weightMatrixIndex > 0) {
		weightMatrixIndex--;
		if (current->next == nullptr) {
			return nullptr;
		}
		current = current->next;
	}
	return current->weights;
}



// -- SET METHODS -- //

// sets every value in the weight matrix to x
void BasicWeight::setAll (double x) {
	BasicWeight* current = this;
	while (current != nullptr) {
		if (current->weights != nullptr) {
			current->weights->setAll(x);
		}
		current = current->next;
	}
}

// sets the value at a specific point to data
void BasicWeight::insertData (float data, int fl, int fw, int fh, int sl, int sw, int sh) {
	long long index = getIndex(fl, fw, fh, sl, sw, sh);
	BasicWeight* current = this;
	while (index >= current->weights->getSize() / sizeof(float)) {
		index -= current->weights->getSize() / sizeof(float);
		current = current->next;
	}
	return current->weights->insert(data, 0, 0, index);
}


// -- GENERATE METHODS -- //

// builds a new weight matrix generating it based on given parameters
// If randomize is 1, randomize using xavier Randomization. 
// if randomize is 2, randomize between -0.5 and 0.5
// else, dont randomize
void BasicWeight::build(int outputSize, int size, int toAdd, int length, int width, int height, int outputLength, int outputWidth, int outputHeight, int randomize) {
	
	// initialize everything
	this->size = size;
	this->outputSize = this->outputSize;
	this->weights = new Matrix3D (1, 1, toAdd);
	this->length = length;
	this->width = width;
	this->height = height;
	this->outputLength = outputLength;
	this->outputWidth = outputWidth;
	this->outputHeight = outputHeight;
	
	// randomize
	if (randomize == 1) this->weights->xavierRandomize(length, width, height, outputLength, outputWidth, outputHeight);
	else if (randomize == 2) this->weights->randomize(-0.5, 0.5);
}

// builds a new weight with the input matrix as the weight matrix
void BasicWeight::build(int outputSize, int size, int length, int width, int height, int outputLength, int outputWidth, int outputHeight, const Matrix3D& inputMatrix) {

	// initialize everything
	this->size = size;
	this->outputSize = outputSize;
	this->weights = new Matrix3D (inputMatrix);
	this->length = length;
	this->width = width;
	this->height = height;
	this->outputLength = outputLength;
	this->outputWidth = outputWidth;
	this->outputHeight = outputHeight;
}


// -- PRINT METHODS -- // 

// prints all of the weights
int BasicWeight::print () {
	this->weights->printMatrix();
	if (this->next != nullptr) {
		this->next->print();
	}
   return 1;
}