//gdb -ex=r --args ./trainGeneralPerceptron.out -a /../data/shroomDataset2/shroom2Key.csv -d /../data/shroomDataset2/data.csv -o /../data/shroomDataset2/shroom2TrainedModel.csv -m 0 -c 4 -e 3 -l 1 -w 10 -h 10 -g 1

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

void loadLine (std::string line, Matrix3D **inputMatrixes, Matrix3D **outputMatrixes, int* index, int outputIndex, int inputSize, int outputSize, std::map<int, std::string> order, std::map<std::string, std::vector<std::string>* >* map);

void help (char *argv[]) {
   fprintf(stderr, "Usage: %s [-g gpu][-e epochs] [-r learningRate] [-l length] [-w width] [-h height] [-c layer count] [-b batch size]  [-d path to data (required if -n not set)] [-n line of csv (required if -d not set)] [-a keyfile (required)] [-o model output file (required)] [-m output index (required)]\n",argv[0]);
   exit(EXIT_FAILURE);
}

int main (int argc, char *argv[]) {
   int startTime = time(0);
   std::cout << std::fixed;
   std::cout.precision(2);


   // get program options
   int opt;

   int epochs = 15;
   int GPU = 0;
   double learningRate = 0.02;
   int hiddenLayerLength = 1; 
   int hiddenLayerWidth = 10;
   int hiddenLayerHeight = 10;
	int batchSize = 16;
   int layerCount = 5;

   std::string inputData = "";
   int outputIndex = -1;

   std::string dataPath = "";
   std::string keyPath = "";
   std::string outputPath = "";
   string* currentPath = new string (filesystem::current_path());
   *currentPath += "/../../../artificialIntelligence/neuralNetworks";
   filesystem::current_path(filesystem::path(*currentPath));

   while ((opt = getopt(argc, argv, "g:e:r:l:w:h:c:b:t:d:a:o:n:m:")) != -1) {
      switch (opt) {
         case 'g':
            GPU = atoi(optarg);
            break;
         case 'e':
            epochs = atoi(optarg);
            break;
         case 'r':
            learningRate = atof(optarg);
            break;
         case 'l':
            hiddenLayerLength = atoi(optarg);
            break;
         case 'w':
            hiddenLayerWidth = atoi(optarg);
            break;
         case 'h':
            hiddenLayerHeight = atoi(optarg);
            break;
         case 'c':
            layerCount = atoi(optarg);
            break;
			case 'b':
            batchSize = atoi(optarg);
            break;
         case 'd':
            dataPath = optarg;
            break;
         case 'a':
            keyPath = optarg;
            break;
         case 'o':
            outputPath = optarg;
            break;
         case 'n':
            inputData = optarg;
            break;
         case 'm':
            outputIndex = atoi(optarg);
            break;
         default: /* '?' */ 
            help(argv);
         }
   }
   if ((dataPath == "" && inputData == "") || keyPath == "" || outputPath == "" || outputIndex == -1) {
      help(argv);
   }

   dataPath = (string) filesystem::current_path() + dataPath;
   keyPath = (string) filesystem::current_path() + keyPath;
   outputPath = (string) filesystem::current_path() + outputPath;
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

   std::cout << "Generating Model Layer 1 of " << layerCount << ".\n";
   LayerList* model = new artificialIntelligence::classes::LayerList ();
   // input layer

   model->addNewBasic (inputMatrixes[0]->getLength(), inputMatrixes[0]->getWidth(), inputMatrixes[0]->getHeight(), ActivationType::Tanh);

   // hidden layers
   for (int i = 0; i < layerCount - 3; i++) {
		std::cout << "Generating Model Layer " << i + 2 << " of " << layerCount << ".\n";
      // BasicLayer* next = new BasicLayer (hiddenLayerLength, hiddenLayerWidth, hiddenLayerHeight);
      // model->add (next);
      model->addNewBasic (hiddenLayerLength, hiddenLayerWidth, hiddenLayerHeight, ActivationType::Tanh);
   }
   std::cout << "Generating Model Layer " << layerCount - 1 << " of " << layerCount << ".\n";
   model->addNewBasic (hiddenLayerLength, hiddenLayerWidth, hiddenLayerHeight, ActivationType::Sigmoid);
	std::cout << "Generating Model Layer " << layerCount << " of " << layerCount << ".\n";
   model->addNewBasic (outputMatrixes[0]->getLength(), outputMatrixes[0]->getWidth(), outputMatrixes[0]->getHeight());
	std::cout << "Model Generated.\n\n";

   model->print(0, 0, 0);
   
   artificialIntelligence::basicLearningTypes::generationalAIBasic::runStochasticGradientDescent(model, epochs, GPU, learningRate, inputMatrixes, outputMatrixes, rows - 1, batchSize, false, false);

   std::cout << "Training Completed.\n\n";
   std::ofstream outputFile;
   outputFile.open (outputPath);
   ((BasicLayer*) model->getRoot())->toFile (&outputFile);
   outputFile.close();
   
   int final = time(0) - startTime;
   std::cout.precision(9);
   std::cout << "\nTime to Complete: " << std::fixed << final << "s\n";
   struct rusage usage;
   getrusage (RUSAGE_SELF, &usage);
   std::cout << "\nMemory used (MB): " << usage.ru_maxrss / 1000000 << "\n\n";
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