//g++ -o TestLayers TestLayers.cpp -O2 -I ../../ --std=c++2a

#include <artificialIntelligence/classes/Basic3DWeightList.cpp>
#include <artificialIntelligence/classes/BasicLayer.cpp>
#include <artificialIntelligence/classes/BasicLayerList.cpp>
#include <artificialIntelligence/classes/BasicWeight.cpp>

#include <coreutils/classes/matrixes/Matrix3D.cpp>

using namespace artificialIntelligence::classes;

void isValidWeight(BasicWeight<float>* testWeight, int flm, int fwm, int fhm, int slm, int swm, int shm, bool print = false);
void testNullWeight ();
void testNotNullWeight ();
void testNullLayer();
void testNotNullLayer (bool print = false);
void testNullLayerList();
void testNotNullLayerList(bool print = false);
void testNotNullLayerList2(bool print = false);
int main () {
   // // Basic weight

   testNullWeight();
   testNotNullWeight();

   // Basic Layer

   testNullLayer();
   testNotNullLayer();

   // BasicLayerList
   testNullLayerList();
   testNotNullLayerList();
   testNotNullLayerList2();
}

void testNullWeight () {
   BasicWeight<float>* testWeight = new BasicWeight<float>();

   if (testWeight->getWeightMatrix(0, 0, 0) != nullptr) {
      std::cout << "testWeight->getWeightMatrix(0, 0, 0) was not nullptr";
      exit (0);
   }
   delete testWeight;
   testWeight = new BasicWeight<float>(1, 1, 1, 1, 1, 1);
   
   if (testWeight->getData(1, 1, 1, 0, 1, 1) != nullptr) {
      std::cout << "testWeight->getData(0, 0, 0, 0, 0, 0) was not nullptr. It was " << testWeight->getData(0, 0, 0, 0, 0, 0);
      exit (0);
   }

   if (testWeight->getWeightMatrix(0, 0, 2) != nullptr) {
      std::cout << "testWeight->getWeightMatrix(0, 0, 2) was not nullptr.";
      exit (0);
   }
   if (testWeight->getData(0, 0, 2, 0, 0, 0) != nullptr) {
      std::cout << "testWeight->getData(0, 0, 2, 0, 0, 0) was not nullptr. It was " << testWeight->getData(0, 0, 2, 0, 0, 0);
      exit (0);
   }

   if (testWeight->getData(0, 0, 2, 0, 0, 2) != nullptr) {
      std::cout << "testWeight->getData(0, 0, 2, 0, 0, 2) was not nullptr. It was " << testWeight->getData(0, 0, 2, 0, 0, 2);
      exit (0);
   }

   std::cout << "BasicWeight :: All weights successfully returned nullptr\n";

   delete testWeight;
}

void testNotNullWeight () {
   BasicWeight<float>* testWeight = new BasicWeight<float>(2, 2, 2, 1, 1, 1);

   if (testWeight->getData(1, 0, 0, 0, 0, 0) == nullptr) {
      std::cout << "testWeight->getData(0, 0, 1, 1, 0, 0) was nullptr.";
      exit (0);
   } 
   delete testWeight;
   
   // testWeight = new BasicWeight<float>(0, 0, 0, 0, 0, 0); errors, which is correct

   // makes a weight from a layer of size 1,1,1 to a layer of size 1,1,1
   testWeight = new BasicWeight<float>(1, 1, 1, 1, 1, 1);
   isValidWeight (testWeight, 1, 1, 1, 1, 1, 1);
   delete testWeight;

   std::cout << "\n\n\n";
   testWeight = new BasicWeight<float>(2, 1, 1, 1, 2, 1);
   isValidWeight (testWeight, 2, 1, 1, 1, 2, 1, true);
   delete testWeight;

   // makes a weight from a layer of size 2,2,2 to a layer of size 1,1,1
   testWeight = new BasicWeight<float>(2, 2, 2, 1, 1, 1);
   isValidWeight (testWeight, 2, 2, 2, 1, 1, 1);
   delete testWeight;

   // makes a weight from a layer of size 1,1,1 to a layer of size 2,2,2
   testWeight = new BasicWeight<float>(1, 1, 1, 2, 2, 2);
   isValidWeight (testWeight, 1, 1, 1, 2, 2, 2);
   delete testWeight;

   // makes a weight from a layer of size 2,2,2 to a layer of size 2,2,2
   testWeight = new BasicWeight<float>(2, 2, 2, 2, 2, 2);
   isValidWeight (testWeight, 2, 2, 2, 2, 2, 2);
   delete testWeight;
   
   // makes a weight from a layer of size 1,2,3 to a layer of size 3,2,1
   testWeight = new BasicWeight<float>(1, 2, 3, 3, 2, 1);
   isValidWeight (testWeight, 1, 2, 3, 3, 2, 1);
   delete testWeight;

   // makes a weight from a layer of size 1,2,3 to a layer of size 3,2,1
   testWeight = new BasicWeight<float>(3, 28, 28, 1, 2, 16);
   isValidWeight (testWeight, 3, 28, 28, 1, 2, 16, true);
   delete testWeight;

   testWeight = new BasicWeight<float>(1, 2, 3, 3, 2, 1);
   isValidWeight (testWeight, 1, 2, 3, 3, 2, 1);
   delete testWeight;

   std::cout << "BasicWeight :: All weights successfully initialized\n";
}

void isValidWeight(BasicWeight<float>* testWeight, int flm, int fwm, int fhm, int slm, int swm, int shm, bool print) {
   if (print) {
      testWeight->print();
   }
   int counter = 0;
   for (int fl = 0; fl < flm; fl++) {
      for (int fw = 0; fw < fwm; fw++) {
         for (int fh = 0; fh < fhm; fh++) {
            for (int sl = 0; sl < slm; sl++) {
               for (int sw = 0; sw < swm; sw++) {
                  for (int sh = 0; sh < shm; sh++) {
                     float* data = testWeight->getData(fl, fw, fh, sl, sw, sh);
                     if (data == nullptr) {
                        std::cout << "data is nullptr";
                        exit (0);
                     }
                     counter++;
                     if (print) {
                        std::cout << "\nValue " << counter << " :: " << *data;
                     }
                  }
               }
            }
         }
      }
   }
   if (print) {
      std::cout << '\n';
   }
}

void testNullLayer() {
   BasicLayer<float>* testLayer = new BasicLayer<float>();

   testLayer->print(false, false);
   testLayer->print(true, false);
   testLayer->print(false, true);
   testLayer->print(true, true);

   testLayer->calculateAndUpdateAll();
   testLayer->print(true, true);

   if (testLayer->getLast() != testLayer) {
      std::cout << "testWeight->getLast() wasnt itself. It was ";
      testLayer->getLast()->print();
      exit (0);
   }

   if (testLayer->getPrev() != nullptr) {
      std::cout << "testWeight->getPrev() wasnt nullptr. It was ";
      testLayer->getPrev()->print();
      exit (0);
   }

   if (testLayer->getNext() != nullptr) {
      std::cout << "testWeight->getNext() wasnt nullptr. It was ";
      testLayer->getNext()->print();
      exit (0);
   }

   std::cout << "BasicWeight :: All layers successfully returned nullptr\n";
}

void testNotNullLayer (bool print) {
   BasicLayer<float>* testLayer = new BasicLayer<float>();
   Matrix3D<float>* layerMatrix = new Matrix3D<float> (1, 1, 1);
   layerMatrix->randomize();
   if (print) {
      layerMatrix->printMatrix();
   }
   testLayer->setLayerMatrix(layerMatrix);
   if (print) {
      testLayer->print();
   }

   delete layerMatrix;
   layerMatrix = new Matrix3D<float> (2, 3, 2);
   layerMatrix->randomize();
   if (print) {
      layerMatrix->printMatrix();
   }
   testLayer->setLayerMatrix(layerMatrix);
   if (print) {
      testLayer->print();
   }
   layerMatrix->insert(5, 0, 0, 0);
   if (print) {
      testLayer->print();
   }

   delete layerMatrix;
   Matrix3D<float>* biasMatrix = new Matrix3D<float> (1, 1, 1);
   biasMatrix->randomize();
   if (print) {
      biasMatrix->printMatrix();
   }
   testLayer->setBiasMatrix(biasMatrix);
   if (print) {
      testLayer->print();
   }

   delete biasMatrix;
   biasMatrix = new Matrix3D<float> (2, 3, 2);
   biasMatrix->randomize();
   if (print) {
      biasMatrix->printMatrix();
   }
   testLayer->setBiasMatrix(biasMatrix);
   if (print) {
      testLayer->print(true);
   }
   biasMatrix->insert(5, 0, 0, 0);
   if (print) {
      testLayer->print(true);
   }

   delete biasMatrix;

   std::cout << "BasicLayer :: All layers successfully initialized";
}

void testNullLayerList() {
   BasicLayerList<float>* list = new BasicLayerList<float> ();

   list->print(false, false);
   list->print(false, true);
   list->print(true, false);
   list->print(true, true);

   list->calculateAndUpdateAll();
   list->calculateAndUpdateLast();

   if (list->getRoot () != nullptr) {
      std::cout << "list->getRoot() wasnt nullptr. It was ";
      list->getRoot ()->print();
   } 
   if (list->getLast () != nullptr) {
      std::cout << "list->getLast() wasnt nullptr. It was ";
      list->getLast ()->print();
   }

   std::cout << "BasicLayerList :: All list components successfully returned nullptr\n";
}

void testNotNullLayerList(bool print) {

   // needs to add test for calculateAndUpdateAll
   BasicLayerList<float>* list = new BasicLayerList<float> ();
   Matrix3D<float>* inputMatrix = new Matrix3D<float>(1, 1, 2);
   Matrix3D<float>* hiddenMatrix = new Matrix3D<float>(1, 1, 2);
   Matrix3D<float>* outputMatrix = new Matrix3D<float>(1, 1, 1);
   
   inputMatrix->insert(1, 0, 0, 0);
   inputMatrix->insert(0.5, 0, 0, 1);
   
   BasicWeight<float>* inputToHiddenWeights = new BasicWeight<float>(1,1,2,1,1,2);
   BasicWeight<float>* hiddenToOutputWeights = new BasicWeight<float>(1,1,2,1,1,1);

   // corresponds to 1
   inputToHiddenWeights->insert(0.72, 0, 0, 0, 0, 0, 0);
   inputToHiddenWeights->insert(0.2, 0, 0, 0, 0, 0, 1);

   // corresponds to 0.5
   inputToHiddenWeights->insert(0.37, 0, 0, 1, 0, 0, 0);
   inputToHiddenWeights->insert(0.64, 0, 0, 1, 0, 0, 1);

   hiddenToOutputWeights->insert(-6, 0, 0, 0, 0, 0, 0);
   hiddenToOutputWeights->insert(5, 0, 0, 1, 0, 0, 0);

   Matrix3D<float>* hiddenBias = new Matrix3D<float> (1, 1, 2);
   Matrix3D<float>* outputBias = new Matrix3D<float> (1, 1, 1);
   
   hiddenBias->insert(1, 0, 0, 0);
   hiddenBias->insert(3, 0, 0, 1);

   outputBias->insert(-1, 0, 0, 0);

   BasicLayer<float>* inputLayer = new BasicLayer<float> (inputMatrix, hiddenBias, inputToHiddenWeights);
   BasicLayer<float>* hiddenLayer = new BasicLayer<float> (hiddenMatrix, outputBias, hiddenToOutputWeights);
   BasicLayer<float>* outputLayer = new BasicLayer<float> (outputMatrix);
   
   list->add(inputLayer);
   list->add(hiddenLayer);
   list->add(outputLayer);
   if (print) {
      list->print(true, true);
   }
   list->calculateAndUpdateAll();
   if (print) {
      list->print(true, true);
   }
   list->calculateAndUpdateAll();
   if (print) {
      list->print(true, true);
   }
   std::cout << "BasicLayerList :: First test successful\n";
}

void testNotNullLayerList2(bool print) {
// needs to add test for calculateAndUpdateAll
   BasicLayerList<float>* list = new BasicLayerList<float> ();
   Matrix3D<float>* inputMatrix = new Matrix3D<float>(1, 1, 2);
   Matrix3D<float>* hiddenMatrix = new Matrix3D<float>(1, 2, 1);
   Matrix3D<float>* outputMatrix = new Matrix3D<float>(1, 1, 1);
   
   inputMatrix->insert(1, 0, 0, 0);
   inputMatrix->insert(0.5, 0, 0, 1);
   
   BasicWeight<float>* inputToHiddenWeights = new BasicWeight<float>(1,1,2,1,2,1);
   BasicWeight<float>* hiddenToOutputWeights = new BasicWeight<float>(1,2,1,1,1,1);

   // exit (0);
   // corresponds to 1
   inputToHiddenWeights->insert(0.72, 0, 0, 0, 0, 0, 0);
   inputToHiddenWeights->insert(0.2, 0, 0, 0, 0, 1, 0);

   // corresponds to 0.5
   inputToHiddenWeights->insert(0.37, 0, 0, 1, 0, 0, 0);
   inputToHiddenWeights->insert(0.64, 0, 0, 1, 0, 1, 0);

   hiddenToOutputWeights->insert(-6, 0, 0, 0, 0, 0, 0);
   hiddenToOutputWeights->insert(5, 0, 1, 0, 0, 0, 0);

   Matrix3D<float>* hiddenBias = new Matrix3D<float> (1, 2, 1);
   Matrix3D<float>* outputBias = new Matrix3D<float> (1, 1, 1);
   
   hiddenBias->insert(1, 0, 0, 0);
   hiddenBias->insert(3, 0, 1, 0);

   outputBias->insert(-1, 0, 0, 0);

   BasicLayer<float>* inputLayer = new BasicLayer<float> (inputMatrix, hiddenBias, inputToHiddenWeights);
   BasicLayer<float>* hiddenLayer = new BasicLayer<float> (hiddenMatrix, outputBias, hiddenToOutputWeights);
   BasicLayer<float>* outputLayer = new BasicLayer<float> (outputMatrix);
   
   list->add(inputLayer);
   list->add(hiddenLayer);
   list->add(outputLayer);
   // list->print(true, true);
   list->calculateAndUpdateAll();
   list->print(true, true);
   list->calculateAndUpdateAll();
   list->print(true, true);

   std::cout << "BasicLayerList :: Second test successful\n";
}