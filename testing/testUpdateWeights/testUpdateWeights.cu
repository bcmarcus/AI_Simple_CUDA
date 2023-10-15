#include <iostream>
#include <unistd.h>

#include <coreutils/util/time.hpp>
#include <coreutils/util/cudaErrors.cuh>

#include <coreutils/classes/matrixes/Tensor.cuh>
#include <coreutils/functions/debug/print.hpp>

#include <artificialIntelligence/classes/weights/BasicWeight.cuh>
#include <artificialIntelligence/classes/layers/BasicLayer.cuh>
#include <artificialIntelligence/classes/layerLists/BasicLayerList.cuh>

using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::debug;

#define LEARNING_RATE 0.1

void test1() {
	int l = 2;
	int w = 2;
	int h = 2;
	Tensor* root = new Tensor(l, w, h);
	root->setAll(1);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 2;
	int w2 = 2;
	int h2 = 2;
	model->addNew(l2, w2, h2);
	model->getRoot()->getNext()->getLayerMatrix()->setAll(1);

	model->getRoot()->getWeights()->setAll(2);
	BasicLayer* currentLayerCPU = model->getRoot();
	BasicLayer* currentLayerGPU = new BasicLayer (*currentLayerCPU, true);
	Tensor* deltaPrevCPU = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrevCPU->setAll(1);
	Tensor* deltaPrevGPU = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrevGPU->setAll(1);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerCPU->updateWeightsCPU(deltaPrevCPU, LEARNING_RATE);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// currentLayerCPU->print(0, 1);
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerGPU->updateWeightsGPU(deltaPrevGPU, LEARNING_RATE);
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	currentLayerCPU->getWeights()->getWeightMatrix(0)->setAll(0);
	std::cout << "EQUALS: " << currentLayerCPU->getWeights()->getWeightMatrix(0)->equals(currentLayerGPU->getWeights()->getWeightMatrix(0)) << '\n';
	// currentLayerGPU->print(0, 1);
	delete model;
	delete deltaPrevCPU;
	delete deltaPrevGPU;
}

void test1random() {
	int l = 2;
	int w = 2;
	int h = 2;
	Tensor* root = new Tensor(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 2;
	int w2 = 2;
	int h2 = 2;
	model->addNew(l2, w2, h2);
	
	
	
	BasicLayer* currentLayerCPU = model->getRoot();
	BasicLayer* currentLayerGPU = new BasicLayer (*currentLayerCPU, true);
	Tensor* deltaPrev = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerCPU->updateWeightsCPU(deltaPrev, LEARNING_RATE);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// currentLayerCPU->print(0, 1);

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerGPU->updateWeightsGPU(deltaPrev, LEARNING_RATE);
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << currentLayerCPU->getWeights()->getWeightMatrix(0)->equals(currentLayerGPU->getWeights()->getWeightMatrix(0)) << '\n';
	// currentLayerGPU->print(0, 1);

	delete model;

}

void test2() {
	int l = 1;
	int w = 2;
	int h = 3;
	Tensor* root = new Tensor(l, w, h);
	root->setAll(1);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 2;
	int w2 = 1;
	int h2 = 2;
	model->addNew(l2, w2, h2);
	model->getRoot()->getNext()->getLayerMatrix()->setAll(1);
	

	model->getRoot()->getWeights()->setAll(2);

	BasicLayer* currentLayerCPU = model->getRoot();
	BasicLayer* currentLayerGPU = new BasicLayer (*currentLayerCPU, true);
	Tensor* deltaPrev = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrev->setAll(1);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerCPU->updateWeightsCPU(deltaPrev, LEARNING_RATE);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerGPU->updateWeightsGPU(deltaPrev, LEARNING_RATE);
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << currentLayerCPU->getWeights()->getWeightMatrix(0)->equals(currentLayerGPU->getWeights()->getWeightMatrix(0)) << '\n';
	// currentLayerCPU->getWeights()->getWeightMatrix(0)->printMatrix();
	// currentLayerGPU->getWeights()->getWeightMatrix(0)->printMatrix();
	delete model;

}

void test2random() {
	int l = 1;
	int w = 2;
	int h = 3;
	Tensor* root = new Tensor(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 2;
	int w2 = 1;
	int h2 = 2;
	model->addNew(l2, w2, h2);
	
	

	BasicLayer* currentLayerCPU = model->getRoot();
	BasicLayer* currentLayerGPU = new BasicLayer (*currentLayerCPU, true);
	Tensor* deltaPrev = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerCPU->updateWeightsCPU(deltaPrev, LEARNING_RATE);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerGPU->updateWeightsGPU(deltaPrev, LEARNING_RATE);
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << currentLayerCPU->getWeights()->getWeightMatrix(0)->equals(currentLayerGPU->getWeights()->getWeightMatrix(0)) << '\n';
	// currentLayerCPU->getWeights()->getWeightMatrix(0)->printMatrix();
	// currentLayerGPU->getWeights()->getWeightMatrix(0)->printMatrix();
	delete model;

}

void test3() {
	int l = 2;
	int w = 2;
	int h = 3;
	Tensor* root = new Tensor(l, w, h);
	root->setAll(1);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 2;
	int w2 = 2;
	int h2 = 3;
	model->addNew(l2, w2, h2);
	model->getRoot()->getNext()->getLayerMatrix()->setAll(1);
	
	

	model->getRoot()->getWeights()->setAll(2);

	BasicLayer* currentLayerCPU = model->getRoot();
	BasicLayer* currentLayerGPU = new BasicLayer (*currentLayerCPU, true);
	Tensor* deltaPrev = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrev->setAll(1);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerCPU->updateWeightsCPU(deltaPrev, LEARNING_RATE);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerGPU->updateWeightsGPU(deltaPrev, LEARNING_RATE);
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << currentLayerCPU->getWeights()->getWeightMatrix(0)->equals(currentLayerGPU->getWeights()->getWeightMatrix(0)) << '\n';
	// currentLayerCPU->getWeights()->getWeightMatrix(0)->printMatrix();
	// currentLayerGPU->getWeights()->getWeightMatrix(0)->printMatrix();
	delete model;

}

void test3random() {
	int l = 2;
	int w = 2;
	int h = 3;
	Tensor* root = new Tensor(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 2;
	int w2 = 2;
	int h2 = 3;
	model->addNew(l2, w2, h2);
	
	

	BasicLayer* currentLayerCPU = model->getRoot();
	BasicLayer* currentLayerGPU = new BasicLayer (*currentLayerCPU, true);
	Tensor* deltaPrev = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerCPU->updateWeightsCPU(deltaPrev, LEARNING_RATE);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerGPU->updateWeightsGPU(deltaPrev, LEARNING_RATE);
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << currentLayerCPU->getWeights()->getWeightMatrix(0)->equals(currentLayerGPU->getWeights()->getWeightMatrix(0)) << '\n';
	// currentLayerCPU->getWeights()->getWeightMatrix(0)->printMatrix();
	// currentLayerGPU->getWeights()->getWeightMatrix(0)->printMatrix();
	delete model;

}

void test4() {
	int l = 10;
	int w = 30;
	int h = 82;
	Tensor* root = new Tensor(l, w, h);
	root->setAll(1);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 6;
	int w2 = 17;
	int h2 = 11;
	model->addNew(l2, w2, h2);
	model->getRoot()->getNext()->getLayerMatrix()->setAll(1);
	

	model->getRoot()->getWeights()->setAll(2);

	BasicLayer* currentLayerCPU = model->getRoot();
	BasicLayer* currentLayerGPU = new BasicLayer (*currentLayerCPU, true);
	Tensor* deltaPrev = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrev->setAll(1);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerCPU->updateWeightsCPU(deltaPrev, LEARNING_RATE);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerGPU->updateWeightsGPU(deltaPrev, LEARNING_RATE);
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << currentLayerCPU->getWeights()->getWeightMatrix(0)->equals(currentLayerGPU->getWeights()->getWeightMatrix(0),  0.00001) << '\n';
	// currentLayerCPU->getWeights()->getWeightMatrix(0)->printMatrix();
	// currentLayerGPU->getWeights()->getWeightMatrix(0)->printMatrix();
	delete model;

}

void test4random() {
	int l = 10;
	int w = 30;
	int h = 82;
	Tensor* root = new Tensor(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 6;
	int w2 = 17;
	int h2 = 11;
	model->addNew(l2, w2, h2);
	
	

	BasicLayer* currentLayerCPU = model->getRoot();
	BasicLayer* currentLayerGPU = new BasicLayer (*currentLayerCPU, true);
	Tensor* deltaPrev = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerCPU->updateWeightsCPU(deltaPrev, LEARNING_RATE);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// currentLayerCPU->print(0, 1);

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerGPU->updateWeightsGPU(deltaPrev, LEARNING_RATE);
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << currentLayerCPU->getWeights()->getWeightMatrix(0)->equals(currentLayerGPU->getWeights()->getWeightMatrix(0),  0.00001) << '\n';
	// currentLayerCPU->print(0, 1);

	delete model;

}

void test45random() {
	int l = 1;
	int w = 5;
	int h = 105;
	Tensor* root = new Tensor(l, w, h);
	root->setAll(1);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 1;
	int w2 = 5;
	int h2 = 25;
	model->addNew(l2, w2, h2);
	model->getRoot()->getNext()->getLayerMatrix()->setAll(1);
	

	// model->getRoot()->getWeights()->setAll(2);
	
	

	BasicLayer* currentLayerCPU = model->getRoot();
	BasicLayer* currentLayerGPU = new BasicLayer (*currentLayerCPU, true);
	Tensor* deltaPrev = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerCPU->updateWeightsCPU(deltaPrev, LEARNING_RATE);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	
	// currentLayerCPU->print(0, 1);
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerGPU->updateWeightsGPU(deltaPrev, LEARNING_RATE);
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	// currentLayerGPU->print(0, 1);
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << currentLayerCPU->getWeights()->getWeightMatrix(0)->equals(currentLayerGPU->getWeights()->getWeightMatrix(0),  0.00001) << '\n';
	// currentLayerCPU->getWeights()->getWeightMatrix(0)->printMatrix();
	// currentLayerGPU->getWeights()->getWeightMatrix(0)->printMatrix();
	delete model;

}


void test5() {
	int l = 3;
	int w = 228;
	int h = 228;
	Tensor* root = new Tensor(l, w, h);
	root->setAll(1);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 1;
	int w2 = 100;
	int h2 = 100;
	model->addNew(l2, w2, h2);
	model->getRoot()->getNext()->getLayerMatrix()->setAll(1);
	

	model->getRoot()->getWeights()->setAll(2);

	BasicLayer* currentLayerCPU = model->getRoot();
	BasicLayer* currentLayerGPU = new BasicLayer (*currentLayerCPU, true);
	Tensor* deltaPrev = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrev->setAll(1);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerCPU->updateWeightsCPU(deltaPrev, LEARNING_RATE);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerGPU->updateWeightsGPU(deltaPrev, LEARNING_RATE);
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << (currentLayerCPU->getWeights()->getWeightMatrix(0)->equals(currentLayerGPU->getWeights()->getWeightMatrix(0),  0.00001) &&
	 									currentLayerCPU->getWeights()->getWeightMatrix(4)->equals(currentLayerGPU->getWeights()->getWeightMatrix(4),  0.00001) &&
										currentLayerCPU->getWeights()->getWeightMatrix(5)->equals(currentLayerGPU->getWeights()->getWeightMatrix(5),  0.00001)) << '\n';
	// currentLayerCPU->getWeights()->getWeightMatrix(0)->printMatrix();
	// currentLayerGPU->getWeights()->getWeightMatrix(23)->printMatrix();
	delete model;

}

void test5random() {
	int l = 3;
	int w = 228;
	int h = 228;
	Tensor* root = new Tensor(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 1;
	int w2 = 100;
	int h2 = 100;
	model->addNew(l2, w2, h2);
	
	

	BasicLayer* currentLayerCPU = model->getRoot();
	BasicLayer* currentLayerGPU = new BasicLayer (*currentLayerCPU, true);
	Tensor* deltaPrev = new Tensor(model->getRoot()->getNext()->getLayerMatrix()->getLength(), model->getRoot()->getNext()->getLayerMatrix()->getWidth(), model->getRoot()->getNext()->getLayerMatrix()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerCPU->updateWeightsCPU(deltaPrev, LEARNING_RATE);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	

	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	currentLayerGPU->updateWeightsGPU(deltaPrev, LEARNING_RATE);
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << (currentLayerCPU->getWeights()->getWeightMatrix(0)->equals(currentLayerGPU->getWeights()->getWeightMatrix(0),  0.00001) &&
	 									currentLayerCPU->getWeights()->getWeightMatrix(4)->equals(currentLayerGPU->getWeights()->getWeightMatrix(4),  0.00001) &&
										currentLayerCPU->getWeights()->getWeightMatrix(5)->equals(currentLayerGPU->getWeights()->getWeightMatrix(5),  0.00001)) << '\n';
	delete model;

}

int main() {
	std::cout << "Starting test 1\n";
	test1();
	std::cout << "\n\n";
	std::cout << "Starting test 1 random\n";
	test1random();
	std::cout << "\n\n";
	std::cout << "Starting test 2\n";
	test2();
	std::cout << "\n\n";
	std::cout << "Starting test 2 random\n";
	test2random();
	std::cout << "\n\n";
	std::cout << "Starting test 3\n";
	test3();
	std::cout << "\n\n";
	std::cout << "Starting test 3 random\n";
	test3random();
	std::cout << "\n\n";
	std::cout << "Starting test 4\n";
	test4();
	std::cout << "\n\n";
	std::cout << "Starting test 4 random\n";
	test4random();
	std::cout << "\n\n";
	std::cout << "Starting test 45 random\n";
	test45random();
	std::cout << "\n\n";
	std::cout << "Starting test 5\n";
	// test5();
	std::cout << "\n\n";
	std::cout << "Starting test 5 random\n";
	test5random();
	std::cout << "\n\n\n";
}