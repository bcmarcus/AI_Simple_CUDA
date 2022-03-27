#include <iostream>
#include <string>
#include <cmath>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "BasicWeight.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

BasicWeight::BasicWeight (int fl, int fw, int fh, int sl, int sw, int sh, bool randomize) {
   this->weights = nullptr;
   this->next = nullptr;
	this->size = fl * fw * fh;
	this->outputSize = sl * sw * sh;
	long long amountLeftToAdd = this->size;
	amountLeftToAdd *= this->outputSize;
	this->length = fl;
	this->width = fw;
	this->height = fh;
	this->outputLength = sl;
	this->outputWidth = sw;
	this->outputHeight = sh;
	this->size = fl * fw * fh * sl * sw * sh;

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

	// want to use BASIC_WEIGHT_MAX_SIZE / size * sizeof(float) as the number of matrixes to combine. to do this, i make the length longer
	BasicWeight* current = this;
	int toAdd = BASIC_WEIGHT_MAX_SIZE;
	if (amountLeftToAdd < BASIC_WEIGHT_MAX_SIZE) {
		toAdd = amountLeftToAdd;
	}
	
	while (amountLeftToAdd > 0) {
		// std::cout << "amountLeftToAdd: " << amountLeftToAdd << "\n";
		// exit(0);
		// std::cout << "amountLeftToAdd: " << amountLeftToAdd << "\n";
		if (amountLeftToAdd < BASIC_WEIGHT_MAX_SIZE) {
			toAdd = amountLeftToAdd;
		}
		current->build(outputSize, this->size, toAdd, length, width, height, outputLength, outputWidth, outputHeight, randomize);
		

		amountLeftToAdd -= toAdd;
		if (amountLeftToAdd > 0) {
			current->next = new BasicWeight();
			current = current->next;
		} else {
			current->next = nullptr;
		}
	}
}
 
void BasicWeight::build(int outputSize, int size, int toAdd, int length, int width, int height, int outputLength, int outputWidth, int outputHeight, bool randomize) {
	this->size = size;
	this->outputSize = this->outputSize;
	// std::cout << "toAdd " << toAdd << '\n';
	this->weights = new Matrix3D (1, 1, toAdd);
	if (randomize) this->weights->xavierRandomize(length, width, height, outputLength, outputWidth, outputHeight);
	this->length = length;
	this->width = width;
	this->height = height;
	this->outputLength = outputLength;
	this->outputWidth = outputWidth;
	this->outputHeight = outputHeight;
}

void BasicWeight::build(int outputSize, int size, int length, int width, int height, int outputLength, int outputWidth, int outputHeight, const Matrix3D& inputMatrix) {
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

// BasicWeight::BasicWeight (int size, int fl, int fh, int fw, int sl, int sw, int sh) {
// 	this->size = size;
// 	this->weights = new Matrix3D(sl, sw, sh);
// 	this->weights->randomize();
// 	this->length = fl;
// 	this->width = fw;
// 	this->height = fh;

// 	this->outputLength = sl;
// 	this->outputWidth = sw;
// 	this->outputHeight = sh;
// }

BasicWeight::BasicWeight () {
	this->size = 0;
	this->weights = nullptr;
	this->next = nullptr;
	this->length = 0;
	this->width = 0;
	this->height = 0;
}

BasicWeight::~BasicWeight(){
   if (this->weights != nullptr) {
		delete this->next;
      delete this->weights;
   } else {
      return;
   }
}

BasicWeight::BasicWeight(const BasicWeight &w){
	BasicWeight* currentCopy = this;
	int currentInputIndex = 0;
	while (w.getWeightMatrix(currentInputIndex) != nullptr) {
		currentCopy->build(w.outputSize, w.size, w.length, w.width, w.height, w.outputLength, w.outputWidth, w.outputHeight, *(w.getWeightMatrix(currentInputIndex)));
		currentCopy->next = new BasicWeight();
		currentInputIndex++;
		if (w.getWeightMatrix(currentInputIndex) != nullptr) {
			currentCopy = currentCopy->next;
		}
	}
	delete currentCopy->next;
	currentCopy->next = nullptr;
}


int BasicWeight::print () {
	this->weights->printMatrix();
	if (this->next != nullptr) {
		this->next->print();
	}
   return 1;
}


int BasicWeight::print (int length, int width, int height) {
	return -1;
}

Matrix3D* BasicWeight::getWeightMatrix (int index) const {
	const BasicWeight* current = this;
	while (index > 0) {
		index--;
		if (current->next == nullptr) {
			return nullptr;
		}
		current = current->next;
	}
	return current->weights;
}

// returns the index of the matrix
long long BasicWeight::getIndex (int fl, int fw, int fh, int sl, int sw, int sh) {
	if (fl > this->length || fw > this->width || fh > this->height) {
		std::cout << "Index out of bounds getIndex in BasicWeights\n";
		exit(0);
	}

	// std::cout << "this->length: " << this->length << "this->width: " << this->width << "this->height: " << this->height << "\n";
	// std::cout << "this->outputWidth: " << this->outputWidth << "this->outputWidth: " << this->outputWidth << "this->outputHeight: " << this->outputHeight << "\n";
	// get the index of the input matrix, multiply it by the outputSize, and then go to the specific spot in the output size
	return (fl * this->width * this->height + fw * this->height + fh) * outputSize  + sl * this->outputWidth * this->outputHeight + sw * this->outputHeight + sh;
}

long long BasicWeight::getSize(){
	return this->size;
}

// goes to the location at that index and returns value shifted by sl sw and sh
float* BasicWeight::getData (int fl, int fw, int fh, int sl, int sw, int sh) {

	// gets the big matrix that the little matrix is actually in
	// std::cout << "fl: " << fl << '\n';
	// std::cout << "fw: " << fw << '\n';
	// std::cout << "fh: " << fh << '\n';
	// std::cout << "sl: " << sl << '\n';
	// std::cout << "sw: " << sw << '\n';
	// std::cout << "sh: " << sh << '\n';
	long long index = getIndex(fl, fw, fh, sl, sw, sh);
	BasicWeight* current = this;
	while (index >= current->weights->getSize() / sizeof(float)) {
		index -= current->weights->getSize() / sizeof(float);
		// std::cout << "indexUpdated: " << index << '\n';
		current = current->next;
	}
	return current->weights->getData(0, 0, index);
}

// goes to the location at that index and inserts value shifted by sl sw and sh
void BasicWeight::insertData (float data, int fl, int fw, int fh, int sl, int sw, int sh) {
	long long index = getIndex(fl, fw, fh, sl, sw, sh);
	BasicWeight* current = this;
	while (index >= current->weights->getSize() / sizeof(float)) {
		index -= current->weights->getSize() / sizeof(float);
		current = current->next;
	}
	return current->weights->insert(data, 0, 0, index);
}


void BasicWeight::setAll (double x) {
	BasicWeight* current = this;
	while (current != nullptr) {
		if (current->weights != nullptr) {
			current->weights->setAll(x);
		}
		current = current->next;
	}
}