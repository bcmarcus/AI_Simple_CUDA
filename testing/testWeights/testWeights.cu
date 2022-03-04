#include <iostream>
#include <unistd.h>

#include <coreutils/util/time.hpp>
#include <coreutils/util/cudaErrors.cuh>

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <coreutils/functions/debug/print.cpp>

#include <artificialIntelligence/classes/BasicWeight.hpp>
#include <artificialIntelligence/classes/BasicLayer.cuh>
#include <artificialIntelligence/classes/BasicLayerList.hpp>

using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::debug;

void test1() {
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

	model->addNew(1, 1, 1);

	model->getRoot()->getLayerMatrix()->setAll(1);
	model->getRoot()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getNext()->getWeights()->setAll(0);

	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPU();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
}

void test2() {
	int l = 2;
	int w = 2;
	int h = 2;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 1;
	int w2 = 2;
	int h2 = 3;
	model->addNew(l2, w2, h2);
	
	int l3 = 2;
	int w3 = 2;
	int h3 = 2;
	model->addNew(l3, w3, h3);

	int l4 = 2;
	int w4 = 2;
	int h4 = 2;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	model->getRoot()->getLayerMatrix()->setAll(1);
	model->getRoot()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getNext()->getWeights()->setAll(0);


	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPU();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
}

void test3() {
	int l = 10;
	int w = 10;
	int h = 10;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 1;
	int w2 = 2;
	int h2 = 3;
	model->addNew(l2, w2, h2);
	
	int l3 = 8;
	int w3 = 12;
	int h3 = 17;
	model->addNew(l3, w3, h3);

	int l4 = 6;
	int w4 = 5;
	int h4 = 4;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	model->getRoot()->getLayerMatrix()->setAll(1);
	model->getRoot()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getNext()->getWeights()->setAll(0);
	

	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPU();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
}

void test4() {
	int l = 10;
	int w = 10;
	int h = 10;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 10;
	int w2 = 10;
	int h2 = 10;
	model->addNew(l2, w2, h2);
	
	int l3 = 20;
	int w3 = 20;
	int h3 = 20;
	model->addNew(l3, w3, h3);

	int l4 = 30;
	int w4 = 30;
	int h4 = 30;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	// model->getRoot()->getLayerMatrix()->randomize();
	model->getRoot()->getLayerMatrix()->setAll(1);
	// model->getRoot()->getWeights()->setAll(0);
	// model->getRoot()->getNext()->getWeights()->setAll(0);
	// model->getRoot()->getNext()->getNext()->getWeights()->setAll(0);
	// model->getRoot()->getNext()->getNext()->getNext()->getWeights()->setAll(0);
	
	double val = 0;
	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPU();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
}

void test5() {
	int l = 3;
	int w = 228;
	int h = 228;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 1;
	int w2 = 100;
	int h2 = 100;
	model->addNew(l2, w2, h2);

	int l3 = 30;
	int w3 = 30;
	int h3 = 30;
	model->addNew(l3, w3, h3);

	int l4 = 10;
	int w4 = 10;
	int h4 = 10;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	model->getRoot()->getLayerMatrix()->randomize();
	// model->getRoot()->getLayerMatrix()->setAll(1);
	// model->getRoot()->getWeights()->setAll(0);
	// model->getRoot()->getNext()->getWeights()->setAll(0);
	// model->getRoot()->getNext()->getNext()->getWeights()->setAll(0);
	// model->getRoot()->getNext()->getNext()->getNext()->getWeights()->setAll(0);
	
	double val = 0;

	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	// model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPU();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
}

void test6() {
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
	
	model->addNew(1, 1, 1);

	model->getRoot()->getLayerMatrix()->setAll(1);
	model->getRoot()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getNext()->getWeights()->setAll(0);

	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
}

void test6random() {
	int l = 2;
	int w = 2;
	int h = 2;
	Matrix3D* root = new Matrix3D(l, w, h);
	root->randomize();
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
	
	model->addNew(1, 1, 1);

	model->print(true, true);

	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
}

void test7() {
	int l = 3;
	int w = 2;
	int h = 1;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 1;
	int w2 = 2;
	int h2 = 3;
	model->addNew(l2, w2, h2);
	
	int l3 = 5;
	int w3 = 5;
	int h3 = 5;
	model->addNew(l3, w3, h3);

	int l4 = 2;
	int w4 = 2;
	int h4 = 2;
	model->addNew(l4, w4, h4);
	
	model->addNew(1, 1, 1);

	model->getRoot()->getLayerMatrix()->setAll(1);
	model->getRoot()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getNext()->getWeights()->setAll(0);

	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
}

void test7random() {
	int l = 3;
	int w = 2;
	int h = 1;
	Matrix3D* root = new Matrix3D(l, w, h);
	root->randomize();
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 1;
	int w2 = 2;
	int h2 = 3;
	model->addNew(l2, w2, h2);
	
	int l3 = 5;
	int w3 = 5;
	int h3 = 5;
	model->addNew(l3, w3, h3);

	int l4 = 2;
	int w4 = 2;
	int h4 = 2;
	model->addNew(l4, w4, h4);
	
	model->addNew(1, 1, 1);
	model->print(true, true);
	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
}

void test8() {
	int l = 10;
	int w = 10;
	int h = 10;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 1;
	int w2 = 2;
	int h2 = 3;
	model->addNew(l2, w2, h2);
	
	int l3 = 8;
	int w3 = 12;
	int h3 = 17;
	model->addNew(l3, w3, h3);

	int l4 = 6;
	int w4 = 5;
	int h4 = 4;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	model->getRoot()->getLayerMatrix()->setAll(1);
	model->getRoot()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getNext()->getWeights()->setAll(0);
	

	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
}


void test8random() {
	int l = 10;
	int w = 10;
	int h = 10;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 1;
	int w2 = 2;
	int h2 = 3;
	model->addNew(l2, w2, h2);
	
	int l3 = 8;
	int w3 = 12;
	int h3 = 17;
	model->addNew(l3, w3, h3);

	int l4 = 6;
	int w4 = 5;
	int h4 = 4;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);

	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
}

void test9() {
	int l = 10;
	int w = 10;
	int h = 10;
	Matrix3D* root = new Matrix3D(l, w, h);
	root->randomize();
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 10;
	int w2 = 10;
	int h2 = 10;
	model->addNew(l2, w2, h2);
	
	int l3 = 20;
	int w3 = 20;
	int h3 = 20;
	model->addNew(l3, w3, h3);

	int l4 = 30;
	int w4 = 30;
	int h4 = 30;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	// model->getRoot()->getLayerMatrix()->randomize();
	model->getRoot()->getLayerMatrix()->setAll(1);
	model->getRoot()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getNext()->getWeights()->setAll(0);
	
	double val = 0;
	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);

	Matrix3D test = Matrix3D(*model->getRoot()->getNext()->getNext()->getNext()->getLayerMatrix());
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	std::cout << "EQUALS: " << test.equals(model->getRoot()->getNext()->getNext()->getNext()->getLayerMatrix()) << '\n';
}


void test9random() {
	std::cout << "Starting test 9\n";
	int l = 10;
	int w = 10;
	int h = 10;
	Matrix3D* root = new Matrix3D(l, w, h);
	root->randomize();
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 10;
	int w2 = 10;
	int h2 = 10;
	model->addNew(l2, w2, h2);
	
	int l3 = 20;
	int w3 = 20;
	int h3 = 20;
	model->addNew(l3, w3, h3);

	int l4 = 10;
	int w4 = 28;
	int h4 = 30;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	std::cout << "Finished loading weights\n";
	double val = 0;
	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	Matrix3D test = Matrix3D(*model->getRoot()->getNext()->getNext()->getNext()->getLayerMatrix());

	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	// test.printMatrix();
	// model->getRoot()->getNext()->getNext()->getNext()->getLayerMatrix()->printMatrix();
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	// std::cout << "EQUALS: " << test.equals(model->getRoot()->getNext()->getLayerMatrix()) << '\n';
}

void test10() {
	int l = 3;
	int w = 228;
	int h = 228;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 1;
	int w2 = 100;
	int h2 = 100;
	model->addNew(l2, w2, h2);

	int l3 = 30;
	int w3 = 30;
	int h3 = 30;
	model->addNew(l3, w3, h3);

	int l4 = 10;
	int w4 = 10;
	int h4 = 10;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	model->getRoot()->getLayerMatrix()->randomize();
	model->getRoot()->getLayerMatrix()->setAll(1);
	model->getRoot()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getNext()->getWeights()->setAll(0);
	
	double val = 0;

	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	Matrix3D test = Matrix3D(*model->getRoot()->getNext()->getNext()->getLayerMatrix());

	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	std::cout << "EQUALS: " << test.equals(model->getRoot()->getNext()->getNext()->getLayerMatrix()) << '\n';
}
void test10random() {
	int l = 3;
	int w = 228;
	int h = 228;
	Matrix3D* root = new Matrix3D(l, w, h);
	root->randomize();
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 1;
	int w2 = 100;
	int h2 = 100;
	model->addNew(l2, w2, h2);

	int l3 = 30;
	int w3 = 30;
	int h3 = 30;
	model->addNew(l3, w3, h3);

	int l4 = 10;
	int w4 = 10;
	int h4 = 10;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	double val = 0;

	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	Matrix3D test = Matrix3D(*model->getRoot()->getNext()->getNext()->getLayerMatrix());

	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	std::cout << "EQUALS: " << test.equals(model->getRoot()->getNext()->getNext()->getLayerMatrix()) << '\n';
}


void test11random() {
	int l = 3;
	int w = 228;
	int h = 228;
	Matrix3D* root = new Matrix3D(l, w, h);
	root->randomize();
	BasicLayerList* model = new BasicLayerList(root);
	int l2 = 5;
	int w2 = 25;
	int h2 = 100;
	model->addNew(l2, w2, h2);
	// std::cout << "here\n";
	// for (int i = 20; i > 10; i--) {
	// 	model->addNew(i, i, i);
	// 	std::cout << "i: " << i << "\n";
	// }

	int l3 = 3;
	int w3 = 30;
	int h3 = 30;
	model->addNew(l3, w3, h3);

	int l4 = 10;
	int w4 = 10;
	int h4 = 10;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	double val = 0;

	sleep(5);
	std::cout << ":::STARTING CPU TESTS:::\n\n\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	Matrix3D test = Matrix3D(*model->getRoot()->getNext()->getNext()->getLayerMatrix());

	sleep(5);
	std::cout << ":::STARTING GPU TESTS:::\n\n\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::Results::::\n\n\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	std::cout << "EQUALS: " << test.equals(model->getRoot()->getNext()->getNext()->getLayerMatrix()) << '\n';
}

void testWeights1() {
	int l = 1;
	int w = 2;
	int h = 3;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 1;
	int w2 = 2;
	int h2 = 3;
	model->addNew(l2, w2, h2);
	model->print(true, true);
	// exit(0);

	std::cout << "\n\n";
	std::cout << "Weights1: " << *model->getRoot()->getWeights()->getData(0, 0, 0, 0, 0, 0) << '\n';
	std::cout << "Weights2: " << *model->getRoot()->getWeights()->getData(0, 0, 0, 0, 1, 2) << '\n';
	std::cout << "Weights3: " << *model->getRoot()->getWeights()->getData(0, 1, 2, 0, 1, 2) << '\n';
	std::cout << "Weights4: " << *model->getRoot()->getWeights()->getData(0, 1, 1, 0, 1, 1) << '\n';
}

void testWeights2() {
	int l = 3;
	int w = 228;
	int h = 228;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 4;
	int w2 = 100;
	int h2 = 100;
	model->addNew(l2, w2, h2);
	
	std::cout << "Weights1: " << *model->getRoot()->getWeights()->getData(0, 0, 0, 0, 0, 0) << '\n';
	std::cout << "Weights2: " << *model->getRoot()->getWeights()->getData(0, 0, 0, 0, 1, 2) << '\n';
	std::cout << "Weights3: " << *model->getRoot()->getWeights()->getData(0, 1, 2, 0, 1, 2) << '\n';
	std::cout << "Weights4: " << *model->getRoot()->getWeights()->getData(0, 1, 1, 0, 1, 1) << '\n';
}

int main() {
	std::cout << "\n\n\n";
	// testWeights1();
	// testWeights2();
	// test1();
	// test2();
	// test3();
	// test4();
	// test5();
	// test6();
	// test6random();
	// test7();
	// test7random();
	// test8();
	// test8random();
	// test9();
	// test9random();
	// test10();
	// test10random();
	test11random();
}