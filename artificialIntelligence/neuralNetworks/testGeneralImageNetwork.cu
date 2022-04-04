// the goal of this is to be able to input a pretrained model, the location of the data to be tested, 
// and the answers for the dataset (all relative paths) either as a file or directory. 
// if it is a directory then go through and test every file in that directory.

// ./testGeneralImageNetwork.out /../data/images/mnist_png/mnistTrainedModelLargeSet4.csv /../data/images/mnist_png/testing/testSelection/four.png /../data/images/mnist_png/testing/answerFile1.txt /../data/images/mnist_png/testing/numberToAnswer.txt
// ./testGeneralImageNetwork.out /../data/images/mnist_png/mnistTrainedModelLargeSet4.csv /../data/images/mnist_png/testing/testSelection/ /../data/images/mnist_png/testing/answerFile1.txt /../data/images/mnist_png/testing/numberToAnswer.txt
// ./testGeneralImageNetwork.out /../data/mnist_png/mnist_png/mnistTrainedModel1.csv /../data/mnist_png/mnist_png/testing/four/four_0.png /../data/mnist_png/mnist_png/indexToAnswerFile.txt

//	./testGeneralImageNetwork.out /../data/dataset/animalTrainedModel1.csv /../data/dataset/dataset/ /../data/dataset/animalAnswerFile.txt


#include <sys/resource.h>
#include <fstream>
#include <filesystem>

#include <coreutils/classes/matrixes/Matrix3D.cuh>

#include <coreutils/functions/sort/sortHelpers.hpp>
#include <coreutils/functions/sort/sortingAlgorithms.hpp>

#include <coreutils/util/time.hpp>

#include <artificialIntelligence/classes/layerLists/BasicLayerList.cuh>

#include <Image_Manipulation/generate/generateInput.hpp>

using namespace std;
using namespace imageEdit;
using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::sort;

bool testSingleImage(std::string dataPath, BasicLayerList* model, std::string* indexToAnswerMap, std::string type, int amountShown = 5);

int testImages(std::string dataPath, BasicLayerList* model, std::string* indexToAnswerMap, std::string type, int* total, int amountShown = 5);

int main (int argc, char **argv) {
   std::cout << std::fixed;
   std::cout << std::setprecision(4);
   
   string* currentPath = new string (filesystem::current_path());
   *currentPath += "/../../../artificialIntelligence/neuralNetworks";
   filesystem::current_path(filesystem::path(*currentPath));

   if (argc < 4) {
      std::cout << "Invalid Number of Arguments\n";
      exit(1);
   }

   string modelPath = (string) filesystem::current_path() + argv[1];
   string dataPath = (string) filesystem::current_path() + argv[2];
   string indexToAnswerPath = (string) filesystem::current_path() + argv[3];
	
   ifstream modelFile (modelPath);
   ifstream dataFile (dataPath);
   ifstream indexToAnswerFile (indexToAnswerPath);
   if (modelFile.good() == false) {
      std::cout << "\n" << (modelPath) << '\n';
      std::cout << "Invalid First Argument\n";
      exit(1);
   }
   if (dataFile.good() == false && std::filesystem::is_regular_file(dataPath)) {
      std::cout << "\n" << dataPath << '\n';
      std::cout << "Invalid Second Argument\n";
      exit(1);
   }
   if (indexToAnswerFile.good() == false) {
      std::cout << "\n" << indexToAnswerPath << '\n';
      std::cout << "Invalid Third Argument\n";
      exit(1);
   }
	
   BasicLayerList* model = new BasicLayerList (modelPath);

   int numOutputs = model->getLast()->getLayer()->getHeight();

   string line;
   std::string *indexToAnswerMap = new string[numOutputs]; 
   int value = 0;
   string name = "";
   while (getline(indexToAnswerFile, line)) {
      value = std::stoi(line.substr(0, line.find(" ")));
      if (line.find(" ") != std::string::npos && line.find(" ") != line.length() - 1) {
         name = line.substr(line.find(" ") + 1, line.length());
      }
      indexToAnswerMap [value] = name;
		std::cout << value << name << "\n";
   }
   
	std::cout << "\n\n\n";
	// exit(0);
   value = 0;
   name = "";

   std::string type;
   if (model->getRoot()->getLayer()->getLength() == 1) {
      type = "BW";
   } else if (model->getRoot()->getLayer()->getLength() == 3) {
      type = "RGB";
   } else if (model->getRoot()->getLayer()->getLength() == 4) {
      type = "RGBA";
   } else {
      std::cout << "Invalid model, invalid input length";
      exit (1);
   }
   
   std::cout << "The current type of image being used is " << type << ".\n";
   
   // check the singular fileaqw
   int correct = 0;
   int* total = new int(0);
   if (std::filesystem::is_regular_file(dataPath)) {
      testSingleImage(dataPath, model, indexToAnswerMap, type, 5);
   } 
   else {
      for (const auto & entry : filesystem::directory_iterator(dataPath)) {
         correct += testImages(entry.path(), model, indexToAnswerMap, type, total, 5);
      }
      std::cout << correct << "/" << *total << " = " << (double) correct / *total * 100 << "%\n\n";
   }
   return 0;
}

int testImages (std::string dataPath, BasicLayerList* model, std::string* indexToAnswerMap, std::string type, int* total, int amountShown) {
   int correct = 0;
   if (dataPath.find(".DS_Store") != std::string::npos) {
      std::cout << "here\n";
      return 0;
   }
   if (std::filesystem::is_regular_file(dataPath)) {
      // std::cout << dataPath << '\n';
      correct += (testSingleImage(dataPath, model, indexToAnswerMap, type, amountShown) ? 1 : 0);
      (*total)++;
   } else {
      for (const auto & entry : filesystem::directory_iterator(dataPath)) {
         correct += (testImages(entry.path(), model, indexToAnswerMap, type, total, amountShown) ? 1 : 0);
      }
   }
   return correct;
}

bool testSingleImage(std::string dataPath, BasicLayerList* model, std::string* indexToAnswerMap, std::string type, int amountShown) {
   Matrix3D* inputMatrix;
   Matrix3D* outputMatrix = new Matrix3D (1, 1, model->getLast()->getLayer()->getHeight());

   int startOfDirectoryIndex = 0;

   
   generate::inputMatrixNormalized(dataPath, &inputMatrix, 0, type);

   int index = -1;
   for (int i = 0; i < model->getLast()->getLayer()->getHeight(); i++) {
      if (dataPath.find(indexToAnswerMap[i], dataPath.find_last_of("/")) != std::string::npos) {
         index = i;
         break;
      }
   }

   if (index == -1) {
      std::cout << "\nInput file " << dataPath << " does not match any answer.\n";
      exit(1);
   }

   outputMatrix->insert(1, 0, 0, index);
   model->setRootMatrix(inputMatrix);

	// tentative before GPU change
	// inputMatrix->printMatrix();
	// model->print(1, 1);
   model->calculateAndUpdateAllGPUV2();
	// model->print(1, 1);
	// exit(0);

   float* topValues = new float[amountShown];
   int* topIndexes = new int[amountShown];
   for (int i = 0; i < amountShown; i++) {
      topValues[i] = 0;
      topIndexes[i] = 0;
   }
   for (int k = 0; k < outputMatrix->getHeight(); k++) {
      if (*(model->getLast()->getLayer()->getData(0, 0, k)) > topValues[0]) {
         topValues[0] = *(model->getLast()->getLayer()->getData(0, 0, k));
         topIndexes[0] = k;
         int* order = insertionSort(topValues, amountShown);
         
         for (int j = 0; j < amountShown; j++) {
            if (order[j] == 0) {
               break;
            }
            swap (topIndexes[j], topIndexes[order[j]]);
         }
      }
   }

   reverse(topValues, amountShown);
   reverse(topIndexes, amountShown);
   std::cout << "\nActual Answer: " << indexToAnswerMap[index] << "\nPredicted Answers: \n";
   for (int i = 0; i < amountShown; i++) {
      std::cout << indexToAnswerMap[topIndexes[i]] << ": " << (topValues[i] * 100) << "%\n";
   }
   std::cout << "\n\n";

   return topIndexes[0] == index;

   // model->getLast()->getLayer()->printMatrix();
   // outputMatrix->printMatrix();
}