#include <iostream>
#include <unistd.h>

#include <coreutils/util/time.hpp>
#include <coreutils/util/cudaErrors.cuh>

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <coreutils/functions/debug/print.hpp>

#include <artificialIntelligence/classes/layerLists/LayerList.cuh>

using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::debug;

#define LEARNING_RATE 0.1

// 4x4 square
void test01() {
	int l = 1;
	int w = 2;
	int h = 2;
	LayerList* model = new LayerList();
	model->addNewConvRoot(l, w, h, 1, 2, 2);
	model->addNewConv(1, 2, 2);
	model->addNewConv(1, 1, 1, 2);
	model->print(true);
}

void test02() {
	int l = 1;
	int w = 2;
	int h = 2;
	LayerList* model = new LayerList();
	model->addNewConvRoot(l, w, h, 1, 2, 2);
	model->addNewConv(1, 2, 2, 2);
	model->addNewConv(1, 3, 3, 4);
	model->print(true, true);
}

void test03() {
	int l = 2;
	int w = 4;
	int h = 4;
	LayerList* model = new LayerList();
	model->addNewConvRoot(l, w, h, 1, 2, 2);
	model->addNewConv(1, 2, 2, 2, 1);
	model->addNewConv(1, 2, 2, 3, 1);
	model->addNewConv(1, 4, 3, 4, 2);
	model->print(true, true, true);
}

void test04() {
	int l = 3;
	int w = 4;
	int h = 4;
	LayerList* model = new LayerList();
	model->addNewConvRoot(l, w, h, 1, 2, 2);
	model->addNewConv(1, 2, 2, 2);
	model->addNewConv(1, 2, 2, 3);
	model->addNewPool(1, 2, 2);
	model->addNewConv(1, 2, 2, 3);
	model->addNewPool(1,2,2);

	model->addNewBasic();
	model->addNewBasic(1,1,8);
	model->addNewBasic(1,1,8);
	model->print(0,0,0);
}

void test05() {
	int l = 1;
	int w = 28;
	int h = 28;
	LayerList* model = new LayerList();
	model->addNewConvRoot(l, w, h, 1, 2, 2);
	model->addNewConv(1, 2, 2, 2, 1);
	model->addNewConv(1, 2, 2, 3, 1);
	model->addNewPool(1,2,2);
	model->addNewConv(1, 2, 2, 3, 1);
	model->addNewPool(1,2,2);
	model->addNewBasic();
	model->addNewBasic(1,20,20);
	model->addNewBasic(1,1,10);
	model->print(0, 0, 0);
}

void test06() {
	int l = 3;
	int w = 128;
	int h = 128;
	LayerList* model = new LayerList();
	model->addNewConvRoot(l, w, h, 1, 3, 3, 64);
	model->addNewConv(1, 3, 3, 32, 1);
	model->addNewConv(1, 3, 3, 16, 1);
	model->addNewPool(1,4,4);
	model->addNewConv(1, 2, 2, 16, 1);
	model->addNewPool(1,2,2);
	model->addNewBasic();
	model->addNewBasic(1,1,150);
	model->addNewBasic(1,1,150);
	model->print(0, 0, 0);
}

void test07() {
	int l = 3;
	int w = 128;
	int h = 128;
	LayerList* model = new LayerList();
	model->addNewConvRoot(l, w, h, 1, 32, 32, 64);
	model->print();
	model->addNewConv(1, 32, 32, 32, 1);
	model->print();
	model->addNewConv(1, 32, 32, 16, 1);
	model->print();
	model->addNewPool(1,2,2);
	model->addNewConv(1, 16, 16, 16, 1);
	model->addNewPool(1,4,4);
	model->addNewBasic();
	model->addNewBasic(1,1,150);
	model->addNewBasic(1,1,150);
	model->print(0, 0, 0);
}

void test1() {
	int l = 1;
	int w = 4;
	int h = 4;
	LayerList* model = new LayerList();
	model->addNewConvRoot(l, w, h, 1, 2, 2);
	model->addNewBasic(l, w, h);
	model->print(0, 0, 0);
	model->getRoot()->getNext()->getLayerMatrix()->setAll(1);

	model->getRoot()->getWeights()->setAll(1);

	std::cout << "\n:::STARTING CPU TESTS:::";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	model->getLast()->getLayerMatrix()->printMatrix();

	std::cout << "\n:::STARTING GPU TESTS:::";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	model->getLast()->getLayerMatrix()->printMatrix();
	

	std::cout << "\n::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	delete model;
}

int main (){
	// std::cout << "Starting test 01\n";
	// test01();
	// std::cout << "\n\n";
	// std::cout << "Starting test 02\n";
	// test02();
	// std::cout << "\n\n";
	// std::cout << "Starting test 03\n";
	// test03();
	// std::cout << "\n\n";
	// std::cout << "Starting test 04\n";
	// test04();
	// std::cout << "\n\n";
	// std::cout << "Starting test 05\n";
	// test05();
	// std::cout << "\n\n";
	// std::cout << "Starting test 06\n";
	// test06();
	// std::cout << "\n\n";
	std::cout << "Starting test 07\n";
	test07();
	std::cout << "\n\n";

	// std::cout << "\n\n\n\n\n";
	// std::cout << "Starting test 1\n";
	// test1();
	// std::cout << "\n\n";
// 	std::cout << "Starting test 1 random\n";
// 	test1random();
// 	std::cout << "\n\n";
// 	std::cout << "Starting test 2\n";
// 	test2();
// 	std::cout << "\n\n";
// 	std::cout << "Starting test 2 random\n";
// 	test2random();
// 	std::cout << "\n\n";
// 	std::cout << "Starting test 3\n";
// 	test3();
// 	std::cout << "\n\n";
// 	std::cout << "Starting test 3 random\n";
// 	test3random();
// 	std::cout << "\n\n";
// 	std::cout << "Starting test 4\n";
// 	test4();
// 	std::cout << "\n\n";
// 	std::cout << "Starting test 4 random\n";
// 	test4random();
// 	std::cout << "\n\n";
// 	std::cout << "Starting test 45 random\n";
// 	test45random();
// 	std::cout << "\n\n";
// 	std::cout << "Starting test 5\n";
// 	// test5();
// 	std::cout << "\n\n";
// 	std::cout << "Starting test 5 random\n";
// 	test5random();
// 	std::cout << "\n\n\n";
}