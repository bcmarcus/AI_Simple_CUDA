#include <iostream>
#include <unistd.h>

#include <coreutils/util/time.hpp>
#include <coreutils/util/cudaErrors.cuh>

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <coreutils/functions/debug/print.hpp>

#include <artificialIntelligence/classes/weights/BasicWeight.cuh>
#include <artificialIntelligence/classes/layers/BasicLayer.cuh>
#include <artificialIntelligence/classes/layerLists/BasicLayerList.cuh>

using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::debug;

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

	model->getRoot()->getLayerMatrix()->randomize();
	model->getRoot()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getWeights()->setAll(0);
	model->getRoot()->getNext()->getNext()->getNext()->getWeights()->setAll(0);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	delete model;
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

	// model->print(true, true);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	delete model;
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

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	delete model;
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
	// model->print(true, true);
	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	delete model;
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

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	delete model;
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

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	double val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	delete model;
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
	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);

	Matrix3D* test = new Matrix3D(*model->getRoot()->getNext()->getNext()->getNext()->getLayerMatrix());
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	std::cout << "EQUALS: " << test->equals(model->getRoot()->getNext()->getNext()->getNext()->getLayerMatrix()) << '\n';
	delete model;
	delete test;
}


void test9random() {
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
	int w4 = 10;
	int h4 = 84;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	double val = 0;
	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	Matrix3D* test = new Matrix3D(*model->getRoot()->getNext()->getNext()->getNext()->getLayerMatrix());

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	// test.printMatrix();
	// model->getRoot()->getNext()->getNext()->getNext()->getLayerMatrix()->printMatrix();
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	std::cout << "EQUALS: " << test->equals(model->getRoot()->getNext()->getLayerMatrix()) << '\n';
	delete model;
	delete test;
}

void test10() {
	int l = 3;
	int w = 228;
	int h = 228;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 10;
	int w2 = 10;
	int h2 = 100;
	model->addNew(l2, w2, h2);

	int l3 = 10;
	int w3 = 10;
	int h3 = 10;
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

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	Matrix3D* test = new Matrix3D(*model->getRoot()->getLast()->getLayerMatrix());

	model->getLast()->getPrev()->print();
	model->getRoot()->getNext()->getLayerMatrix()->setAll(0);
	model->getRoot()->getNext()->getNext()->getLayerMatrix()->setAll(0);
	model->getRoot()->getNext()->getNext()->getNext()->getLayerMatrix()->setAll(0);

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	model->getLast()->getPrev()->print();
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	std::cout << "EQUALS: " << test->equals(model->getRoot()->getLast()->getLayerMatrix()) << '\n';
	delete model;
	delete test;
}
void test10random() {
	int l = 3;
	int w = 228;
	int h = 228;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);

	int l2 = 10;
	int w2 = 10;
	int h2 = 100;
	model->addNew(l2, w2, h2);

	int l3 = 10;
	int w3 = 10;
	int h3 = 10;
	model->addNew(l3, w3, h3);

	int l4 = 10;
	int w4 = 10;
	int h4 = 10;
	model->addNew(l4, w4, h4);

	model->addNew(1, 1, 1);
	
	model->getRoot()->getLayerMatrix()->randomize();
	
	double val = 0;

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	Matrix3D* test = new Matrix3D(*model->getRoot()->getLast()->getLayerMatrix());

	model->getLast()->getPrev()->print();
	model->getRoot()->getNext()->getLayerMatrix()->setAll(0);
	model->getRoot()->getNext()->getNext()->getLayerMatrix()->setAll(0);
	model->getRoot()->getNext()->getNext()->getNext()->getLayerMatrix()->setAll(0);

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	model->getLast()->getPrev()->print();
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	std::cout << "EQUALS: " << test->equals(model->getRoot()->getLast()->getLayerMatrix()) << '\n';
	delete model;
	delete test;
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
	
	for (int i = 20; i > 10; i--) {
		model->addNew(i, i, i);
		std::cout << "i: " << i << "\n";
	}

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

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllCPU();
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayerMatrix()->insert(-17, 0, 0, 0);
	Matrix3D* test = new Matrix3D(*model->getRoot()->getLast()->getLayerMatrix());

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	model->calculateAndUpdateAllGPUV2();
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "CPU ANSWER: " << val << '\n';

	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "GPU ANSWER: " << *model->getLast()->getLayerMatrix()->getData(0, 0, 0) << '\n';
	std::cout << "EQUALS: " << test->equals(model->getRoot()->getLast()->getLayerMatrix()) << '\n';
	delete model;
	delete test;
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
	// model->print(true, true);
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
	std::cout << "Starting test 6\n";
	test6();
	std::cout << "\n\n";
	std::cout << "Starting test 6 random\n";
	test6random();
	std::cout << "\n\n";
	std::cout << "Starting test 7\n";
	test7();
	std::cout << "\n\n";
	std::cout << "Starting test 7 random\n";
	test7random();
	std::cout << "\n\n";
	std::cout << "Starting test 8\n";
	test8();
	std::cout << "\n\n";
	std::cout << "Starting test 8 random\n";
	test8random();
	std::cout << "\n\n";
	std::cout << "Starting test 9\n";
	test9();
	std::cout << "\n\n";
	std::cout << "Starting test 9 random\n";
	test9random();
	std::cout << "\n\n";
	std::cout << "Starting test 10\n";
	test10();
	std::cout << "\n\n";
	std::cout << "Starting test 10 random\n";
	test10random();
	std::cout << "\n\n";
	std::cout << "Starting test 11 random\n";
	test11random();
	std::cout << "\n\n";
}