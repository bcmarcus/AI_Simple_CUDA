#include <iostream>
#include <coreutils/util/time.hpp>

#include <coreutils/classes/matrixes/Matrix3D.cpp>
#include <coreutils/functions/debug/print.cpp>

#include <artificialIntelligence/classes/BasicWeight.hpp>
#include <artificialIntelligence/classes/BasicLayer.hpp>
#include <artificialIntelligence/classes/BasicLayerList.hpp>

using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::debug;

// gpu error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int main() {
	std::cout << "Hello world\n";
	int l = 2;
	int w = 2;
	int h = 2;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 2;
	int w2 = 2;
	int h2 = 2;
	model->addNew(l2, w2, h2);
	
	int l3 = 2;
	int w3 = 2;
	int h3 = 2;
	model->addNew(l3, w3, h3);

	int l4 = 2;
	int w4 = 2;
	int h4 = 2;
	model->addNew(l4, w4, h4);

	model->print();
}