#include <iostream>
#include <random>
#include <cmath>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include <coreutils/util/time.hpp>
#include <coreutils/util/cudaErrors.cuh>
#include <coreutils/functions/debug/print.cpp>
#include <coreutils/functions/sort/sortHelpers.cpp>
#include <coreutils/functions/math/simpleMath.hpp>

using namespace coreutils::functions;
using namespace coreutils::classes::matrixes;

int Matrix3D::getLength () {
	return this->length;
}

int Matrix3D::getWidth () {
	return this->width;
}

int Matrix3D::getHeight () {
	return this->height;
}

float* Matrix3D::getArr () {
	return arr;
}

long long Matrix3D::getSize () {
	return memorySize;
}

float* Matrix3D::getData (int length, int width, int height) {
	if (this->length <= length || this->width <= width || this->height <= height) {
		std::cout << "Invalid input at getData";
		return nullptr;
	}
	return &this->arr[getIndex(length, width, height)];
}

int Matrix3D::getIndex (int l, int w, int h) {
	return l * this->width * this->height + w * this->height + h;
}

// shuffles every single value
void Matrix3D::shuffleEvery () {
	srand(GetTimeStamp().tv_sec + GetTimeStamp().tv_usec);
	for (int length = 0; length < this->length; length++) {
		for (int width = 0; width < this->width; width++) {
			for (int height = 0; height < this->height; height++) {
				sort::swap (&this->arr[getIndex(length, width, height)], 
								&this->arr[getIndex((double) rand() / RAND_MAX * length, (double) rand() / RAND_MAX * width, (double) rand() / RAND_MAX * height)]);

			}
		}
	}
}


// shuffles every 2d matrix. while retaining the 2d matrix
int* Matrix3D::shuffleGroups () {
	int* order = new int[this->length];
	for (int length = 0; length < this->length; length++) {
		// int randomLength = (int) math::rand(0, this->length - 1);
		srand(GetTimeStamp().tv_sec + GetTimeStamp().tv_usec);
		double randomLength = rand() / RAND_MAX * length;
		order[length] = randomLength;
		for (int width = 0; width < this->width; width++) {
			for (int height = 0; height < this->height; height++) {
				sort::swap (&this->arr [getIndex(length, width, height)], 
								&this->arr [getIndex(randomLength, width, height)]);

			}
		}
	}
	return order;
}

void Matrix3D::shuffleGroups (int* order) {
	for (int length = 0; length < this->length; length++) {
		for (int width = 0; width < this->width; width++) {
			for (int height = 0; height < this->height; height++) {
				sort::swap (&this->arr [getIndex(length, width, height)], 
								&this->arr [getIndex(order[length], width, height)]);

			}
		}
	}
}

// -- -- //

// adds this and another matrix and 
// sets this matrix equal to it
void Matrix3D::operator += (const Matrix3D* m2) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				this->arr [getIndex(i, j, k)] += m2->arr [getIndex(i, j, k)];
			}
		}
	}
}

// subtracts this and another matrix and 
// sets this matrix equal to it 
void Matrix3D::operator -= (const Matrix3D* m2) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				this->arr [getIndex(i, j, k)] -= m2->arr [getIndex(i, j, k)];
			}
		}
	}
}


// returns addition of this and another matrix
Matrix3D* Matrix3D::operator + (const Matrix3D* m2) {
	Matrix3D* M3D = new Matrix3D (this->length, this->width, this->height);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				M3D->arr[getIndex(i, j, k)] = this->arr [getIndex(i, j, k)] + m2->arr [getIndex(i, j, k)];
			}
		}
	}

	return M3D;
}

// returns subtraction of this and another matrix
Matrix3D* Matrix3D::operator - (const Matrix3D* m2) {
	Matrix3D* M3D = new Matrix3D (this->length, this->width, this->height);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				M3D->arr[getIndex(i, j, k)] = this->arr [getIndex(i, j, k)] - m2->arr [getIndex(i, j, k)];
			}
		}
	}

	return M3D;
}

// returns multiplication of this and another matrix
Matrix3D* Matrix3D::operator * (const Matrix3D* m2) {
	Matrix3D* M3D = new Matrix3D(this->length, this->width, this->height);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				M3D->arr[getIndex(i, j, k)] = this->arr [getIndex(i, j, k)] * m2->arr [getIndex(i, j, k)];
			}
		}
	}

	return M3D;
}

Matrix3D* Matrix3D::operator * (const float x) {
	Matrix3D* M3D = new Matrix3D (this->length, this->width, this->height);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				M3D->arr[getIndex(i, j, k)] = this->arr [getIndex(i, j, k)] * x;
			}
		}
	}
	
	return M3D;
}

Matrix3D* Matrix3D::operator / (const Matrix3D* m2) {
	Matrix3D* M3D = new Matrix3D (this->length, this->width, this->height);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				M3D->arr[getIndex(i,j,k)] = this->arr [getIndex(i,j,k)] / m2->arr [getIndex(i,j,k)];
			}
		}
	}
	
	return M3D;
}

bool Matrix3D::equals (const Matrix3D* m2) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				if (m2->arr[getIndex(i, j, k)] != this->arr [getIndex(i, j, k)]) {
					return false;
				}
			}
		}
	}

	return true;
}

void Matrix3D::randomize (double lowerBound, double upperBound) {
	double currentRandomNumber;
	srand(GetTimeStamp().tv_sec + GetTimeStamp().tv_usec);
	for (int i = 0; i < this->length; i++) {
		for (int j = 0; j < this->width; j++) {
			for (int k = 0; k < this->height; k++) {
				currentRandomNumber = ((double) rand()) / RAND_MAX * (upperBound - lowerBound) + lowerBound;
				this->arr [getIndex(i, j, k)] = currentRandomNumber;
			}
		}
	}
}

double Matrix3D::dotProduct (const Matrix3D* m2) {
	double output = 0;
	
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				output += this->arr [getIndex(i, j, k)] * m2->arr [getIndex(i, j, k)];
			}
		}
	}
	
	return output;
}

double Matrix3D::sum () {
	double output = 0;
	
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				output += this->arr [getIndex(i, j, k)];
			}
		}
	}
	
	return output;
}

void Matrix3D::insert (float data, int length, int width, int height) {
	// printf("[%d][%d][%d] should be %d but is %d\n", length, width, height, length * this->width * this->height + width * this->height + height, getIndex(length, width, height));
	this->arr[getIndex(length, width, height)] = data;
}

void Matrix3D::printMatrix () {
	std::cout << '\n' << "{";
	for (int i = 0; i < this->length; i++) {
		std::cout << '\n' << "  {" << '\n';
		for (int j = 0; j < this->width; j++) {
			std::string out = "    {";
			for (int k = 0; k < this->height; k++) {
				out += std::to_string(this->arr [getIndex(i, j, k)]) + ", ";
			}
			out = out.substr(0, out.length () - 2);
			std::cout << out << "}" << '\n';
		}
		std::cout << "  }";
	}
	std::cout << '\n' << "}" << '\n';
}

void Matrix3D::setMatrix (Matrix3D* M3D) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				this->insert(*M3D->getData(i, j, k), i, j, k);
			}
		}
	}
}

void Matrix3D::setAll (double x) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				this->insert(x, i, j, k);
			}
		}
	}
}

// Matrix3D findAndMakeMatrix (std::ifstream in) {
//    std::string line;
//    int length = 0;
//    int width = 0;
//    int height = 0;
//    while (getline (in, line)) {
//       // if ()
//    }
// }

Matrix3D::Matrix3D (const int length, const int width, const int height) {
	// std::cout << "Constructor\n";
	this->length = length;
	this->width = width;
	this->height = height;
	gpuErrchk(cudaMallocHost((void **) &this->arr, length * width * height * sizeof(float)));
	// this->arr = new float [length * width * height];
	this->memorySize = length * width * height * sizeof(float);
}

Matrix3D::Matrix3D (const Matrix3D &m3d) {
	this->length = m3d.length;
	this->width = m3d.width;
	this->height = m3d.height;
	gpuErrchk(cudaMallocHost((void **) &this->arr, length * width * height * sizeof(float)));
	gpuErrchk(cudaMemcpy(this->arr, m3d.arr, length * width * height * sizeof(float), cudaMemcpyHostToHost));
	this->memorySize = length * width * height * sizeof(float);
}

Matrix3D::Matrix3D () {
	// std::cout << "Constructor\n";
	this->length = 0;
	this->width = 0;
	this->height = 0;
	this->arr = nullptr;
	this->memorySize = 0;
}

Matrix3D::~Matrix3D () {
	cudaFreeHost(this->arr);
}