#include <iostream>
#include <fstream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "../layerLists/BasicLayerList.cuh"
#include "../layers/BasicLayer.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;


// -- CONSTRUCTOR DESTRUCTOR COPY -- //

// default constructor
BasicLayerList::BasicLayerList () {
   this->root = nullptr;
   this->last = nullptr;
}

// constructor
BasicLayerList::BasicLayerList (Matrix3D* layer, Matrix3D* biasMatrix, BasicWeight* weights){
   this->root = new BasicLayer (layer, biasMatrix, weights);
   this->last = this->root;
}

// loads a model from a file
BasicLayerList::BasicLayerList (std::string filepath) {
   std::ifstream inputFile (filepath);
   if (inputFile.good() == false) {
      exit(0);
   }
   this->root = BasicLayer::loadFromFile(&inputFile);
   this->last = ((BasicLayer*) this->root)->getLast();
   inputFile.close();
}

// destructor
BasicLayerList::~BasicLayerList() {
	delete this->root;
}

// copy constructor
BasicLayerList::BasicLayerList(const BasicLayerList& bll) {
	this->root = new BasicLayer(*(BasicLayer* ) bll.root);
	this->last = ((BasicLayer*) (this->root))->getLast();
}


// -- GET METHODS -- //

// gets the root layer
BasicLayer* BasicLayerList::getRoot () {
   return ((BasicLayer*) (this->root));
}

// gets the last layer
BasicLayer* BasicLayerList::getLast () {
   return ((BasicLayer*) (this->last));
}


// -- SET METHODS -- //

// sets the root matrix
void BasicLayerList::setRootMatrix (Matrix3D* newMatrix) {
   if (this->root != nullptr) {
      this->root->setLayer(newMatrix);
   }
}

// -- GENERATE METHODS -- //

// adds a layer at the end of the model recursively 
void BasicLayerList::add (BasicLayer* layer) {
   if (this->last != nullptr) {
      this->last = ((BasicLayer*) (this->last))->add(layer);
   } else {
      this->root = layer;
      this->last = this->root;
   }
}

// creates and adds a layer at the end of the model recursively
void BasicLayerList::add (Matrix3D* layerMatrix, Matrix3D* biasMatrix, BasicWeight* weights) {
   if (this->root == nullptr) {
      this->root = new BasicLayer (layerMatrix, biasMatrix, weights);
   } else {
      this->root = ((BasicLayer*) (this->root))->add(layerMatrix, biasMatrix, weights);
   }
   this->last = ((BasicLayer*) (this->root))->getLast();
}

// creates and adds a layer at the end of the model recursively
void BasicLayerList::addNew (int length, int width, int height) {
   Matrix3D* layerMatrix = new Matrix3D (length, width, height);
   layerMatrix->randomize();
   if (this->root == nullptr) {
      this->root = new BasicLayer (layerMatrix);
   } else {
      ((BasicLayer*) (this->root))->add(layerMatrix);
   }
   this->last = ((BasicLayer*) (this->root))->getLast();
}


// -- LAYER UPDATE METHODS -- //
            
// updates all layers in the model using CPU compute
void BasicLayerList::calculateAndUpdateAllCPU () {
   if (this->root != nullptr) {
      ((BasicLayer*) (this->root))->calculateAndUpdateAllCPU();
   } else {
      std::cout << "No root layer initialized!\n";
   }
}

// updates the last layer using CPU compute
void BasicLayerList::calculateAndUpdateLastCPU () {
   if (this->last != nullptr) {
      ((BasicLayer*) (this->last))->getPrev()->calculateAndUpdateAllCPU();
   } else {
      std::cout << "No last layer initialized!\n";
   }
}

// updates all layers in the model using GPU compute revised
void BasicLayerList::calculateAndUpdateAllGPUV2 () {
   if (this->root != nullptr) {
      ((BasicLayer*) (this->root))->calculateAndUpdateAllGPUV2();
   } else {
      std::cout << "No root layer initialized!\n";
   }
}


// -- PRINT METHODS -- //

// prints the entire model
void BasicLayerList::print (bool printBias, bool printWeights) {
   if (this->root != nullptr) {
      int depth = ((BasicLayer*) (this->root))->print(printBias, printWeights);
      std::cout << "There are " << depth << " total layers\n";
   } else {
      std::cout << "No root layer initialized!\n";
   }
}

// loads a model into a file using the format of 
// layer length, layer width, layer height
// bias length, bias width, bias height
// <the values for the bias, all comma seperated>
// layer length, layer width, layer height, bias length, bias width, bias height
// <the values for the weights, with each float16 represented by 4 bytes of data> 
void BasicLayerList::toFile (std::string filepath) {
   std::ofstream outputFile;
   outputFile.open (filepath);
   ((BasicLayer*) (this->root))->toFile (&outputFile);
   outputFile.close();
}