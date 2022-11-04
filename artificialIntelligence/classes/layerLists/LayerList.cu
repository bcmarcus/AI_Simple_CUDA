#include <iostream>
#include <fstream>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include "../layerLists/LayerList.cuh"
#include "../layers/ConvLayer.cuh"
#include "../layers/PoolLayer.cuh"
#include "../layers/BasicLayer.cuh"

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;


// -- CONSTRUCTOR DESTRUCTOR COPY -- //

// default constructor
LayerList::LayerList () {
   this->root = nullptr;
   this->last = nullptr;
   this->totalParamCount = 0;
}

LayerList::LayerList (LayerBase* layer) {
   this->root = layer;
   this->last = this->root->getLast();
   this->totalParamCount = layer->paramCount ();
}

// // constructor
// LayerList::LayerList (Matrix3D* layer, Matrix3D* biasMatrix, WeightBase* weights){
//    this->root = new LayerList (layer, biasMatrix, weights);
//    this->last = this->root;
// }

// loads a model from a file
LayerList::LayerList (std::string filepath) {
   // std::ifstream inputFile (filepath);
   // if (inputFile.good() == false) {
      // exit(0);
   // }
   // this->root = LayerBase::loadFromFile(&inputFile);
   // this->last = ((LayerBase*) this->root)->getLast();
   // inputFile.close();
}

// destructor
LayerList::~LayerList() {
	delete this->root;
}

// // copy constructor
// LayerList::LayerList(const LayerList& cll) {
// 	this->root = new LayerBase(*(LayerBase* ) cll.root);
// 	this->last = ((LayerBase*) (this->root))->getLast();
// }


// -- GET METHODS -- //

// gets the root layer
LayerBase* LayerList::getRoot () {
   return ((LayerBase*) (this->root));
}

// gets the last layer
LayerBase* LayerList::getLast () {
   return ((LayerBase*) (this->last));
}


// -- SET METHODS -- //

// sets the root matrix
void LayerList::copyRootMatrix (Matrix3D* newMatrix) {
   if (this->root != nullptr) {
      this->root->copyLayerMatrix(newMatrix);
   }
}

// -- GENERATE METHODS -- //

// adds a layer at the end of the model recursively 
void LayerList::add (LayerBase* layer, int index) {
   if (this->last != nullptr) {
      this->last = this->last->add (layer, index);
   } else {
      this->root = layer;
      this->last = this->root;
   }
}

void LayerList::addNewBasic (int length, int width, int height, ActivationType activationType, int index) {
   BasicLayer* next = new BasicLayer (length, width, height, activationType);
   if (this->root == nullptr) {
      this->root = next;
      this->last = next;
   } else {
      this->root->add (next, index);
      this->last = this->root->getLast();
   }
}

// creates and adds a layer at the end of the model recursively
void LayerList::addNewPool (int poolLength, int poolWidth, int poolHeight, ActivationType activationType, int index) {
   if (this->root == nullptr) {
      std::cout << "Pooling layer cannot be first layer\n";
      exit (1);
   } else {
      PoolLayer* next = new PoolLayer 
      (
         this->root->getLast()->getLayerMatrix()->getLength(), 
         this->root->getLast()->getLayerMatrix()->getWidth(), 
         this->root->getLast()->getLayerMatrix()->getHeight(), 
         poolLength,
         poolWidth, 
         poolHeight,
         activationType
      );
      this->root->add (next, index);
      this->last = this->root->getLast();
   }
}

// creates and adds a layer at the end of the model recursively
void LayerList::addNewConv (int length, int width, int height, int convLength, int convWidth, int convHeight, int features, int stride, ActivationType activationType, int index) {
   ConvLayer* next = new ConvLayer (length, width, height, convLength, convWidth, convHeight, features, stride, activationType);
   if (this->root == nullptr) {
      this->root = next;
      this->last = next;
   } else {
      this->root->add (next, index);
      this->last = this->root->getLast();
   }
}


// -- LAYER UPDATE METHODS -- //
            
// updates all layers in the model using CPU compute
void LayerList::calculateAndUpdateAllCPU () {
   if (this->root != nullptr) {
      this->root->calculateAndUpdateAllCPU();
   } else {
      std::cout << "No root layer initialized!\n";
   }
}

// updates the last layer using CPU compute
void LayerList::calculateAndUpdateLastCPU () {
   if (this->last != nullptr) {
      this->last->getPrev()->calculateAndUpdateAllCPU();
   } else {
      std::cout << "No last layer initialized!\n";
   }
}

// updates all layers in the model using GPU compute revised
void LayerList::calculateAndUpdateAllGPUV2 () {
   if (this->root != nullptr) {
      this->root->calculateAndUpdateAllGPUV2();
   } else {
      std::cout << "No root layer initialized!\n";
   }
}


// -- PRINT METHODS -- //

// prints the entire model
void LayerList::print (bool printLayer, bool printBias, bool printWeights) {
   if (this->root != nullptr) {
      this->root->print(printLayer, printBias, printWeights);
      std::cout << "\n\nThere are " << this->totalParamCount << " total parameters\n";
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
void LayerList::toFile (std::string filepath) {
   std::ofstream outputFile;
   outputFile.open (filepath);
   this->root->toFile (&outputFile);
   outputFile.close();
}

LayerList* LayerList::loadFromFile (std::string filepath) {
   LayerList* list;

   std::ifstream* inputFile = new std::ifstream(filepath);
   if (inputFile->good() == false) {
      exit(0);
   }

   std::string line;
   getline (*inputFile, line);
   switch (stoi(line)) {
      case LayerBase::LayerType::Basic:
         list = new LayerList (BasicLayer::loadFromFile (inputFile, nullptr));
         break;
      case LayerBase::LayerType::Conv:
         list = new LayerList (ConvLayer::loadFromFile (inputFile, nullptr));
         break;
      case LayerBase::LayerType::Pool:
         list = new LayerList (PoolLayer::loadFromFile (inputFile, nullptr));
         break;
   }
   list->last = list->root;
   
   while (!inputFile->eof()) {
      LayerBase* current = list->root->getLast();
      getline (*inputFile, line);

      switch (stoi(line)) {
         case LayerBase::LayerType::Basic:
            current->setNext (BasicLayer::loadFromFile (inputFile, current), 0);
            break;
         case LayerBase::LayerType::Conv:
            current->setNext (ConvLayer::loadFromFile (inputFile, current), 0);
            break;
         case LayerBase::LayerType::Pool:
            current->setNext (PoolLayer::loadFromFile (inputFile, current), 0);
            break;
      }
   }
   
   list->last = list->root->getLast();
   inputFile->close();

   return list;
}