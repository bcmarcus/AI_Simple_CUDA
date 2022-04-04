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
	
	model->addNew(1, 1, 1);

	model->getRoot()->getWeights()->setAll(2);

	BasicLayer* currentLayer = model->getRoot();
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	Matrix3D* deltaPrev = new Matrix3D(model->getRoot()->getNext()->getLayer()->getLength(), model->getRoot()->getNext()->getLayer()->getWidth(), model->getRoot()->getNext()->getLayer()->getHeight());
	deltaPrev->setAll(1);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	double val;
	error = currentLayer->calculateErrorCPU(deltaPrev);
	// error->printMatrix(); 


	Matrix3D* tempError = new Matrix3D(*error);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayer()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	error = model->getRoot()->calculateErrorGPU(deltaPrev);
	// error->printMatrix(); 
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << tempError->equals(error) << '\n';
	std::cout << *tempError->getData(l - 1, w - 1, h - 1) << '\n';
	std::cout << *error->getData(l - 1, w - 1, h - 1);
	delete model;
	delete tempError;
	delete error;
}

void test1random() {
	int l = 2;
	int w = 2;
	int h = 2;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 2;
	int w2 = 2;
	int h2 = 2;
	model->addNew(l2, w2, h2);
	
	model->addNew(1, 1, 1);

	BasicLayer* currentLayer = model->getRoot();
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	Matrix3D* deltaPrev = new Matrix3D(model->getRoot()->getNext()->getLayer()->getLength(), model->getRoot()->getNext()->getLayer()->getWidth(), model->getRoot()->getNext()->getLayer()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	double val;

	error = currentLayer->calculateErrorCPU(deltaPrev);


	Matrix3D* tempError = new Matrix3D(*error);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayer()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	error = model->getRoot()->calculateErrorGPU(deltaPrev);
	// error->printMatrix(); 
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << tempError->equals(error) << '\n';
	std::cout << *tempError->getData(l - 1, w - 1, h - 1) << '\n';
	std::cout << *error->getData(l - 1, w - 1, h - 1);
	delete model;
	delete tempError;
	delete error;
}

void test2() {
	int l = 1;
	int w = 2;
	int h = 3;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 2;
	int w2 = 1;
	int h2 = 2;
	model->addNew(l2, w2, h2);
	
	model->addNew(1, 1, 1);

	model->getRoot()->getWeights()->setAll(2);

	BasicLayer* currentLayer = model->getRoot();
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	Matrix3D* deltaPrev = new Matrix3D(model->getRoot()->getNext()->getLayer()->getLength(), model->getRoot()->getNext()->getLayer()->getWidth(), model->getRoot()->getNext()->getLayer()->getHeight());
	deltaPrev->setAll(1);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	double val;
	error = currentLayer->calculateErrorCPU(deltaPrev);
	// error->printMatrix(); 

	Matrix3D* tempError = new Matrix3D(*error);

	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayer()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	error = model->getRoot()->calculateErrorGPU(deltaPrev);
	// error->printMatrix(); 
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << tempError->equals(error) << '\n';
	std::cout << *tempError->getData(l - 1, w - 1, h - 1) << '\n';
	std::cout << *error->getData(l - 1, w - 1, h - 1);
	delete model;
	delete tempError;
	delete error;
}

void test2random() {
	int l = 1;
	int w = 2;
	int h = 3;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 2;
	int w2 = 1;
	int h2 = 2;
	model->addNew(l2, w2, h2);
	
	model->addNew(1, 1, 1);

	BasicLayer* currentLayer = model->getRoot();
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	Matrix3D* deltaPrev = new Matrix3D(model->getRoot()->getNext()->getLayer()->getLength(), model->getRoot()->getNext()->getLayer()->getWidth(), model->getRoot()->getNext()->getLayer()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	double val;

	error = currentLayer->calculateErrorCPU(deltaPrev);
	// error->printMatrix(); 



	Matrix3D* tempError = new Matrix3D(*error);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayer()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	error = model->getRoot()->calculateErrorGPU(deltaPrev);
	// error->printMatrix(); 
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << tempError->equals(error) << '\n';
	std::cout << *tempError->getData(l - 1, w - 1, h - 1) << '\n';
	std::cout << *error->getData(l - 1, w - 1, h - 1);
	delete model;
	delete tempError;
	delete error;
}

void test3() {
	int l = 2;
	int w = 2;
	int h = 3;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 2;
	int w2 = 2;
	int h2 = 3;
	model->addNew(l2, w2, h2);
	
	model->addNew(1, 1, 1);

	model->getRoot()->getWeights()->setAll(2);

	BasicLayer* currentLayer = model->getRoot();
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	Matrix3D* deltaPrev = new Matrix3D(model->getRoot()->getNext()->getLayer()->getLength(), model->getRoot()->getNext()->getLayer()->getWidth(), model->getRoot()->getNext()->getLayer()->getHeight());
	deltaPrev->setAll(1);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	double val;
	error = currentLayer->calculateErrorCPU(deltaPrev);
	// error->printMatrix(); 


	Matrix3D* tempError = new Matrix3D(*error);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayer()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	error = model->getRoot()->calculateErrorGPU(deltaPrev);
	// error->printMatrix(); 
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;

	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << tempError->equals(error) << '\n';
	std::cout << *tempError->getData(l - 1, w - 1, h - 1) << '\n';
	std::cout << *error->getData(l - 1, w - 1, h - 1);
	delete model;
	delete tempError;
	delete error;
}

void test3random() {
	int l = 2;
	int w = 2;
	int h = 3;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 2;
	int w2 = 2;
	int h2 = 3;
	model->addNew(l2, w2, h2);
	
	model->addNew(1, 1, 1);

	BasicLayer* currentLayer = model->getRoot();
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	Matrix3D* deltaPrev = new Matrix3D(model->getRoot()->getNext()->getLayer()->getLength(), model->getRoot()->getNext()->getLayer()->getWidth(), model->getRoot()->getNext()->getLayer()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	double val;

	error = currentLayer->calculateErrorCPU(deltaPrev);
	// error->printMatrix(); 



	Matrix3D* tempError = new Matrix3D(*error);
	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayer()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	error = model->getRoot()->calculateErrorGPU(deltaPrev);
	// error->printMatrix(); 
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << tempError->equals(error) << '\n';
	std::cout << *tempError->getData(l - 1, w - 1, h - 1) << '\n';
	std::cout << *error->getData(l - 1, w - 1, h - 1);
	delete model;
	delete tempError;
	delete error;
}

void test4() {
	int l = 10;
	int w = 10;
	int h = 82;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 6;
	int w2 = 17;
	int h2 = 11;
	model->addNew(l2, w2, h2);
	
	model->addNew(1, 1, 1);

	model->getRoot()->getWeights()->setAll(2);

	BasicLayer* currentLayer = model->getRoot();
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	Matrix3D* deltaPrev = new Matrix3D(model->getRoot()->getNext()->getLayer()->getLength(), model->getRoot()->getNext()->getLayer()->getWidth(), model->getRoot()->getNext()->getLayer()->getHeight());
	deltaPrev->setAll(1);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	double val;
	error = currentLayer->calculateErrorCPU(deltaPrev);
	// error->printMatrix(); 

	Matrix3D* tempError = new Matrix3D(*error);
	std::cout << "\n\n";


	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayer()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	error = model->getRoot()->calculateErrorGPU(deltaPrev);
	// error->printMatrix(); 
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << tempError->equals(error) << '\n';
	std::cout << *tempError->getData(l - 1, w - 1, h - 1) << '\n';
	std::cout << *error->getData(l - 1, w - 1, h - 1);
	delete model;
	delete tempError;
	delete error;
}

void test4random() {
	int l = 10;
	int w = 30;
	int h = 82;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 6;
	int w2 = 17;
	int h2 = 11;
	model->addNew(l2, w2, h2);
	
	model->addNew(1, 1, 1);

	BasicLayer* currentLayer = model->getRoot();
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	Matrix3D* deltaPrev = new Matrix3D(model->getRoot()->getNext()->getLayer()->getLength(), model->getRoot()->getNext()->getLayer()->getWidth(), model->getRoot()->getNext()->getLayer()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	double val;

	error = currentLayer->calculateErrorCPU(deltaPrev);
	// error->printMatrix(); 

	Matrix3D* tempError = new Matrix3D(*error);
	std::cout << "\n\n";


	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayer()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	error = model->getRoot()->calculateErrorGPU(deltaPrev);
	// error->printMatrix(); 
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << tempError->equals(error) << '\n';
	std::cout << *tempError->getData(l - 1, w - 1, h - 1) << '\n';
	std::cout << *error->getData(l - 1, w - 1, h - 1);
	delete model;
	delete tempError;
	delete error;
}

void test45random() {
	int l = 3;
	int w = 228;
	int h = 228;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 100;
	int w2 = 1;
	int h2 = 1;
	model->addNew(l2, w2, h2);
	
	model->addNew(1, 1, 1);

	BasicLayer* currentLayer = model->getRoot();
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	Matrix3D* deltaPrev = new Matrix3D(model->getRoot()->getNext()->getLayer()->getLength(), model->getRoot()->getNext()->getLayer()->getWidth(), model->getRoot()->getNext()->getLayer()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	double val;
	
	error = currentLayer->calculateErrorCPU(deltaPrev);
	// error->printMatrix(); 
	
	Matrix3D* tempError = new Matrix3D(*error);
	std::cout << "\n\n";


	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayer()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	error = model->getRoot()->calculateErrorGPU(deltaPrev);
	// error->printMatrix(); 
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << tempError->equals(error) << '\n';
	std::cout << *tempError->getData(l - 1, w - 1, h - 1) << '\n';
	std::cout << *error->getData(l - 1, w - 1, h - 1);
	delete model;
	delete tempError;
	delete error;
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
	
	model->addNew(1, 1, 1);

	model->getRoot()->getWeights()->setAll(2);

	BasicLayer* currentLayer = model->getRoot();
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	Matrix3D* deltaPrev = new Matrix3D(model->getRoot()->getNext()->getLayer()->getLength(), model->getRoot()->getNext()->getLayer()->getWidth(), model->getRoot()->getNext()->getLayer()->getHeight());
	deltaPrev->setAll(1);

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	double val;
	error = currentLayer->calculateErrorCPU(deltaPrev);
	// error->printMatrix(); 

	Matrix3D* tempError = new Matrix3D(*error);
	std::cout << "\n\n";


	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayer()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	error = model->getRoot()->calculateErrorGPU(deltaPrev);
	// error->printMatrix(); 
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << tempError->equals(error) << '\n';
	std::cout << *tempError->getData(l - 1, w - 1, h - 1) << '\n';
	std::cout << *error->getData(l - 1, w - 1, h - 1);
	delete model;
	delete tempError;
	delete error;
}

void test5random() {
	int l = 3;
	int w = 228;
	int h = 228;
	Matrix3D* root = new Matrix3D(l, w, h);
	BasicLayerList* model = new BasicLayerList(root);
	
	int l2 = 1;
	int w2 = 100;
	int h2 = 100;
	model->addNew(l2, w2, h2);
	
	model->addNew(1, 1, 1);

	BasicLayer* currentLayer = model->getRoot();
	Matrix3D* currentLayerMatrix = currentLayer->getLayer();
	Matrix3D* error = new Matrix3D(currentLayerMatrix->getLength(), currentLayerMatrix->getWidth(), currentLayerMatrix->getHeight());
	Matrix3D* deltaPrev = new Matrix3D(model->getRoot()->getNext()->getLayer()->getLength(), model->getRoot()->getNext()->getLayer()->getWidth(), model->getRoot()->getNext()->getLayer()->getHeight());
	deltaPrev->randomize();

	std::cout << ":::STARTING CPU TESTS:::\n";
	double cpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	double val;

	error = currentLayer->calculateErrorCPU(deltaPrev);
	// error->printMatrix(); 
	// exit(0);
	Matrix3D* tempError = new Matrix3D(*error);
	std::cout << "\n\n";


	double cpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - cpuStartTime;
	// val = *model->getLast()->getLayerMatrix()->getData(0, 0, 0);
	model->getLast()->getLayer()->insert(-17, 0, 0, 0);
	
	std::cout << ":::STARTING GPU TESTS:::\n";
	double gpuStartTime = GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000;
	error = model->getRoot()->calculateErrorGPU(deltaPrev);
	// error->printMatrix(); 
	double gpuFinalTime = (GetTimeStamp().tv_sec + (double) GetTimeStamp().tv_usec / 1000000) - gpuStartTime;
	
	std::cout << "::::RESULTS::::\n";
	std::cout << "CPU TIME TO COMPLETE: " << cpuFinalTime << '\n';
	std::cout << "GPU TIME TO COMPLETE: " << gpuFinalTime << '\n';
	std::cout << "EQUALS: " << tempError->equals(error) << '\n';
	std::cout << *tempError->getData(l - 1, w - 1, h - 1) << '\n';
	std::cout << *error->getData(l - 1, w - 1, h - 1);
	delete model;
	delete tempError;
	delete error;
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
	test5();
	std::cout << "\n\n";
	std::cout << "Starting test 5 random\n";
	test5random();
	std::cout << "\n\n\n";
}