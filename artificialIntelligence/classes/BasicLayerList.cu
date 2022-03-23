#include <iostream>
#include <fstream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "BasicLayerList.hpp"
#include "BasicLayer.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;

BasicLayerList::BasicLayerList (Matrix3D* layer, Matrix3D* biasMatrix, BasicWeight* weights){
   this->root = new BasicLayer (layer, biasMatrix, weights);
   this->last = this->root;
}


BasicLayerList::BasicLayerList () {
   this->root = nullptr;
   this->last = nullptr;
}

BasicLayerList::~BasicLayerList() {
	delete this->root;
}

void BasicLayerList::print (bool printBias, bool printWeights) {
   if (this->root != nullptr) {
      int depth = this->root->print(printBias, printWeights);
      std::cout << "There are " << depth << " total layers\n";
   } else {
      std::cout << "No root layer initialized!\n";
   }
}


void BasicLayerList::add (Matrix3D* layerMatrix, Matrix3D* biasMatrix, BasicWeight* weights) {
   if (this->root == nullptr) {
      this->root = new BasicLayer (layerMatrix, biasMatrix, weights);
   } else {
      this->root = this->root->add(layerMatrix, biasMatrix, weights);
   }
   this->last = this->root->getLast();
}


void BasicLayerList::editRootMatrix (Matrix3D* newMatrix) {
   if (this->root != nullptr) {
      this->root->setLayerMatrix(newMatrix);
   }
}

// void BasicLayerList::calculateAndUpdateAllGPU () {
//    if (this->root != nullptr) {
//       this->root->calculateAndUpdateAllGPU();
//    } else {
//       std::cout << "No root layer initialized!\n";
//    }
// }

void BasicLayerList::calculateAndUpdateAllGPUV2 () {
   if (this->root != nullptr) {
      this->root->calculateAndUpdateAllGPUV2();
   } else {
      std::cout << "No root layer initialized!\n";
   }
}

void BasicLayerList::calculateAndUpdateAllCPU () {
   if (this->root != nullptr) {
      this->root->calculateAndUpdateAllCPU();
   } else {
      std::cout << "No root layer initialized!\n";
   }
}


void BasicLayerList::calculateAndUpdateLastCPU () {
   if (this->last != nullptr) {
      this->last->getPrev()->calculateAndUpdateAllCPU();
   } else {
      std::cout << "No last layer initialized!\n";
   }
}


void BasicLayerList::add (BasicLayer* layer) {
   if (this->last != nullptr) {
      this->last = this->last->add(layer);
   } else {
      this->root = layer;
      this->last = this->root;
   }
}


void BasicLayerList::addNew (int length, int width, int height) {
   Matrix3D* layerMatrix = new Matrix3D (length, width, height);
   layerMatrix->randomize();

   // Matrix3D* biasMatrix = new Matrix3D (length, width, height);
   // biasMatrix->randomize();
   if (this->root == nullptr) {
      this->root = new BasicLayer (layerMatrix);
   } else {
      this->root->add(layerMatrix);
   }
   this->last = this->root->getLast();
}


BasicLayer* BasicLayerList::getRoot () {
   return this->root;
}


BasicLayer* BasicLayerList::getLast () {
   return this->last;
}


void BasicLayerList::toFile (std::string filepath) {
   std::ofstream outputFile;
   outputFile.open (filepath);
   this->root->toFile (&outputFile);
   outputFile.close();
}


BasicLayerList* BasicLayerList::loadFromFile (std::string filepath) {
   BasicLayerList* list = new BasicLayerList();
   std::ifstream inputFile (filepath);
   if (inputFile.good() == false) {
      return nullptr;
   }
   list->root = BasicLayer::loadFromFile(&inputFile);
   list->last = list->root->getLast();
   inputFile.close();
   return list;
}