//gdb -ex=r --args ./testGeneralPerceptron.out -a /../data/shroomDataset2/shroom2Key.csv -d /../data/shroomDataset2/data.csv -o /../data/shroomDataset2/shroom2TrainedModel.csv -m 0 -g 1

#include <sys/time.h>
#include <sys/resource.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <map>

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <coreutils/functions/sort/sortHelpers.hpp>
#include <coreutils/functions/sort/sortingAlgorithms.hpp>

#include <artificialIntelligence/basicLearningTypes/generationalAIBasic.cuh>
#include <artificialIntelligence/classes/layerLists/LayerList.cuh>
#include <artificialIntelligence/classes/layers/LayerBase.cuh>
#include <artificialIntelligence/classes/layers/BasicLayer.cuh>

using namespace std;
using namespace coreutils::classes::matrixes;
using namespace coreutils::functions::sort;

void loadLine (std::string line, Matrix3D **inputMatrixes, Matrix3D **outputMatrixes, int* index, int outputIndex, int inputSize, int outputSize, std::map<int, std::string> order, std::map<std::string, std::vector<std::string>* >* map);

void help (char *argv[]) {
   fprintf(stderr, "Usage: %s [-g gpu] [-d path to data (required if -n not set)] [-n line of data (required if -d not set)] [-a keyfile (required)] [-o model output file (required)] [-m output index (required)]\n",argv[0]);
   exit(EXIT_FAILURE);
}

int main (int argc, char *argv[]) {
   int startTime = time(0);
   std::cout << std::fixed;
   std::cout.precision(4);

   // get program options
   int opt;

   int GPU = 0;
   int amountShown = 2;
   int num = -1;

   std::string inputData = "";
   int outputIndex = -1;

   std::string dataPath = "";
   std::string keyPath = "";
   std::string modelPath = "";
   string* currentPath = new string (filesystem::current_path());
   *currentPath += "/../../../artificialIntelligence/neuralNetworks";
   filesystem::current_path(filesystem::path(*currentPath));

   while ((opt = getopt(argc, argv, "g:d:a:o:n:m:c:")) != -1) {
      switch (opt) {
         case 'g':
            GPU = atoi(optarg);
            break;
         case 'd':
            dataPath = optarg;
            break;
         case 'a':
            keyPath = optarg;
            break;
         case 'o':
            modelPath = optarg;
            break;
         case 'n':
            inputData = optarg;
            break;
         case 'm':
            outputIndex = atoi(optarg);
            break;
         case 'c':
            num = atoi(optarg);
            break;
         default: /* '?' */ 
            help(argv);
         }
   }
   if ((dataPath == "" && inputData == "") || keyPath == "" || modelPath == "" || outputIndex == -1) {
      help(argv);
   }

   dataPath = (string) filesystem::current_path() + dataPath;
   keyPath = (string) filesystem::current_path() + keyPath;
   modelPath = (string) filesystem::current_path() + modelPath;
   delete currentPath;

   ifstream dataFile (dataPath);
   ifstream keyFile (keyPath);

   if (dataFile.good() == false && inputData == "") {
      std::cout << "\n" << dataPath << '\n';
      std::cout << inputData << "\n";
      std::cout << "Invalid Data Path or Data\n";
      exit(1);
   }
   if (keyFile.good() == false) {
      std::cout << "\n" << keyPath << '\n';
      std::cout << "Invalid Key Path\n";
      exit(1);
   }

   // -- CREATE INPUT AND OUTPUT MATRIX ARRAYS START -- //
   
   int rows = 0;

   // -- CREATE INPUT AND OUTPUT MATRIX ARRAYS END-- //
   std::string line;
   getline (dataFile, line);
   if (dataFile.good ()) {
      while (getline (dataFile, line)) {
         rows++;
      }
      dataFile.clear(); 
      dataFile.seekg(0, std::ios::beg);
   } else {
      rows = 1;
   }
   
   Matrix3D** inputMatrixes = new Matrix3D* [rows];
   Matrix3D** outputMatrixes = new Matrix3D* [rows];

   // -- LOAD KEY FILE START -- //

   int inputSize = 0;
   int outputSize = 0;

   // get the first line and see how many columns there are
   std::string line2;
   std::stringstream lineStream;

   std::string value;
   std::map<int, std::string> order;
   std::map<std::string, std::vector<std::string>* >* map = new std::map<std::string, std::vector<std::string>* > ();
   
   int columns = 0;
   while (getline (keyFile, line)) {
      if (!getline (keyFile, line2)) {
         break;
      }
   
      std::stringstream lineStream;
      lineStream << line2;
      map->insert (std::pair<std::string,std::vector<std::string>* >(line, new std::vector<std::string> ()));
      
      while (getline(lineStream, value, ',')) {
         if (columns == outputIndex) {
            outputSize++;
         } else {
            inputSize++;
         }
         map->at (line)->push_back (value);
      }
   

      order.insert (std::pair<int, std::string> (columns, line));
      columns++;
   }
   
   for (auto it = order.begin(); it != order.end(); ++it) {
      vector <std::string>* v = map->at (it->second);
      std::cout << it->second << "\n";
      for (int k = 0; k < v->size() - 1; k++) {
         std::cout << v->at(k) << ", ";
      }
      std::cout << v->at(v->size() - 1);
      std::cout << "\n";
   }

   // -- LOAD KEY FILE END -- //


   // -- LOAD LINES INTO MATRIXES START -- //

   int* index = new int(0);
   if (dataFile.good ()) {
      dataFile.clear(); 
      dataFile.seekg(0, std::ios::beg);
      getline (dataFile, line);
      while (getline (dataFile, line)) {
         loadLine (line, inputMatrixes, outputMatrixes, index, outputIndex, inputSize, outputSize, order, map);
         (*index)++;
      }
   } else {
      loadLine (inputData, inputMatrixes, outputMatrixes, index, outputIndex, inputSize, outputSize, order, map);
   }

   // inputMatrixes[0]->printMatrix();
   // inputMatrixes[1]->printMatrix();
   // outputMatrixes[0]->printMatrix();
   // outputMatrixes[1]->printMatrix();




   // -- LOAD LINES INTO MATRIXES END -- //

   LayerList* model = LayerList::loadFromFile (modelPath);
   
	std::cout << "Model Loaded.\n\n";
   model->print(0, 0, 0);
   
   // std::map <int, std::string> indexToAnswerMap;
   std::vector<std::string>* indexToAnswerMap = map->at(order.at(outputIndex));
   // for (int i = 0, cc = v->size(); i < cc; i++) {
      // indexToAnswerMap.insert(std::pair<int, std::string> (i, v->at(i)));
   // }
   if (num != -1) {
      rows = num;
   }

   float currentError = 0;
   int correct = 0;
   for (int i = 0, cc2 = rows; i < cc2; i++) {
      int index = -1;
      for (int j = 0, cc3 = outputMatrixes[0]->getHeight(); j < cc3; j++) {
         if (*outputMatrixes[i]->getData(0, 0, j) != 0) {
            index = j;
         }
      }

      if (index == -1) {
         std::cout << "\nOutput broken\n";
         exit(1);
      }


      model->copyRootMatrix(inputMatrixes[i]);
      if (GPU) model->calculateAndUpdateAllGPUV2();
      else model->calculateAndUpdateAllCPU();

      Matrix3D* error = *outputMatrixes[i] - model->getLast()->getLayerMatrix();
      Matrix3D* squared = *error * error;
      // std::cout << "4\n";	
      currentError += squared->sum() * 100;
      delete error;
      delete squared;

      float* topValues = new float[amountShown];
      int* topIndexes = new int[amountShown];
      for (int q = 0; q < amountShown; q++) {
         topValues[q] = 0;
         topIndexes[q] = 0;
      }

      for (int k = 0; k < outputMatrixes[i]->getHeight(); k++) {
         if (*(model->getLast()->getLayerMatrix()->getData(0, 0, k)) > topValues[0]) {
            topValues[0] = *(model->getLast()->getLayerMatrix()->getData(0, 0, k));
            topIndexes[0] = k;
            int* order = insertionSort(topValues, amountShown);
            for (int j = 0; j < amountShown; j++) {
               // std::cout << j << '\n';
               if (order[j] == 0) {
                  break;
               }
               swap (topIndexes[j], topIndexes[order[j]]);
            }
         }
      }

      reverse(topValues, amountShown);
      reverse(topIndexes, amountShown);
      std::cout << "\nActual Answer: " << indexToAnswerMap->at(index) << "\nPredicted Answers: \n";
      for (int z = 0; z < amountShown; z++) {
         std::cout << indexToAnswerMap->at(topIndexes[z]) << ": " << (topValues[z] * 100) << "%\n";
      }
      std::cout << "\n\n";

      correct += (topIndexes[0] == index);
   }
						
   std::cout << "STATS\n";
   std::cout << "Total error:                 " << currentError << "%\n\n";
   std::cout << "Random Guessing Accuracy:    " << 1 / (double) outputMatrixes[0]->getHeight() << '\n';
   std::cout << "Predicted Accuracy:" << correct << "/" << rows << " :: " << correct / (double) rows << "\n";
}

void loadLine (std::string line, Matrix3D **inputMatrixes, Matrix3D **outputMatrixes, int* index, int outputIndex, int inputSize, int outputSize, std::map<int, std::string> order, std::map<std::string, std::vector<std::string>* >* map) {
   Matrix3D* im = new Matrix3D (1, 1, inputSize);
   Matrix3D* om = new Matrix3D (1, 1, outputSize);

   inputMatrixes [*index] = im;
   outputMatrixes [*index] = om;
   std::stringstream linestream;
   std::string value;
   linestream << line;
   int inputIndex = 0;
   int counter = 0;
   int spot = -1;
   while (getline (linestream, value, ',')) {
      spot = -1;
      std::vector<std::string>* v = map->at(order.at(counter));
      for (int k = 0, cc = v->size (); k < cc; k++) {
         if (value == v->at (k)) {
            spot = k;
         }
      }
      if (spot == -1) {
         std::cout << "Input Data not found in Key Mapping File :: " << "Key: " << order.at (counter) << "  Value: " << value << '\n';
         exit (1);
      }

      if (counter == outputIndex) {
         om->insert(1, 0, 0, spot);
      } else {
         im->insert (1, 0, 0, inputIndex + spot);
         inputIndex += v->size ();
      }
      counter++;
   }
}