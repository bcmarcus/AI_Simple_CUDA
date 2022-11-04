#include <iostream>
#include <unistd.h>

#include <coreutils/util/time.hpp>
#include <coreutils/util/cudaErrors.cuh>

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <coreutils/functions/debug/print.hpp>

#include <artificialIntelligence/classes/weights/BasicWeight.cuh>
#include <artificialIntelligence/classes/layers/BasicLayer.cuh>
#include <artificialIntelligence/classes/layerLists/LayerList.cuh>

using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::debug;

void test1() {
	int l = 2;
	int w = 2;
	int h = 2;
	LayerList* model = new LayerList ();

	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);

	
	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	Matrix3D* cpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix(0)));

	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	Matrix3D* gpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix()));

	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << cpuMat->equals(gpuMat) << '\n';
	std::cout << *cpuMat->getData(0, 0, 0) << '\n';
	std::cout << *gpuMat->getData(0, 0, 0);
	delete model;
	delete cpuMat;
	delete gpuMat;
}

void test2() {
	int l = 1;
	int w = 10;
	int h = 10;
	LayerList* model = new LayerList ();

	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);

	
	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	Matrix3D* cpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix(0)));

	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	Matrix3D* gpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix()));

	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << cpuMat->equals(gpuMat) << '\n';
	std::cout << *cpuMat->getData(0, 0, 0) << '\n';
	std::cout << *gpuMat->getData(0, 0, 0);
	delete model;
	delete cpuMat;
	delete gpuMat;
}

void test3() {
	int l = 10;
	int w = 10;
	int h = 10;
	LayerList* model = new LayerList ();

	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	
	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	Matrix3D* cpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix(0)));

	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	Matrix3D* gpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix()));

	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << cpuMat->equals(gpuMat) << '\n';
	std::cout << *cpuMat->getData(0, 0, 0) << '\n';
	std::cout << *gpuMat->getData(0, 0, 0);
	delete model;
	delete cpuMat;
	delete gpuMat;
}

void test4() {
	int l = 10;
	int w = 10;
	int h = 10;
	LayerList* model = new LayerList ();

	model->addNewBasic(l, w, h, ActivationType::Tanh);
	model->addNewBasic(l, w, h, ActivationType::Tanh);
	model->addNewBasic(l, w, h, ActivationType::Tanh);
	model->addNewBasic(l, w, h, ActivationType::Tanh);
	model->addNewBasic(l, w, h, ActivationType::Tanh);
	
	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	Matrix3D* cpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix(0)));

	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	Matrix3D* gpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix()));

	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << cpuMat->equals(gpuMat) << '\n';
	std::cout << *cpuMat->getData(0, 0, 0) << '\n';
	std::cout << *gpuMat->getData(0, 0, 0);
	delete model;
	delete cpuMat;
	delete gpuMat;
}

void test5() {
	int l = 2;
	int w = 2;
	int h = 2;
	LayerList* model = new LayerList ();

	model->addNewBasic(l, w, h, ActivationType::LeakyRelu);
	model->addNewBasic(l, w, h, ActivationType::LeakyRelu);
	model->addNewBasic(l, w, h, ActivationType::LeakyRelu);
	model->addNewBasic(l, w, h, ActivationType::LeakyRelu);
	model->addNewBasic(l, w, h, ActivationType::LeakyRelu);
	
	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	Matrix3D* cpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix(0)));

	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	Matrix3D* gpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix()));

	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << cpuMat->equals(gpuMat) << '\n';
	std::cout << *cpuMat->getData(0, 0, 0) << '\n';
	std::cout << *gpuMat->getData(0, 0, 0);
	delete model;
	delete cpuMat;
	delete gpuMat;
}

void test6() {
	int l = 10;
	int w = 10;
	int h = 10;
	LayerList* model = new LayerList ();

	model->addNewBasic(l, w, h, ActivationType::Tanh);
	model->addNewBasic(l, w, h, ActivationType::Tanh);
	model->addNewBasic(l, w, h, ActivationType::Tanh);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	model->addNewBasic(l, w, h, ActivationType::Sigmoid);
	
	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	Matrix3D* cpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix(0)));

	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	Matrix3D* gpuMat = new Matrix3D (*(model->getLast()->getLayerMatrix()));

	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << cpuMat->equals(gpuMat) << '\n';
	std::cout << *cpuMat->getData(0, 0, 0) << '\n';
	std::cout << *gpuMat->getData(0, 0, 0);
	delete model;
	delete cpuMat;
	delete gpuMat;
}

int main() {
	std::cout << "Starting test 1\n";
	test1();
	std::cout << "\n\n";
	std::cout << "Starting test 2\n";
	test2();
	std::cout << "\n\n";
	std::cout << "Starting test 3\n";
	test3();
	std::cout << "\n\n";
	std::cout << "Starting test 4\n";
	test4();
	std::cout << "\n\n";
	std::cout << "Starting test 5\n";
	test5();
	std::cout << "\n\n";
	std::cout << "Starting test 6\n";
	test6();
	std::cout << "\n\n";
}