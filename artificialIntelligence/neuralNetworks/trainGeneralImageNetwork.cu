//    ./trainGeneralImageNetwork.out -t RGB -d /../data/dataset/dataset -a /../data/dataset/animalAnswerFile.txt -o /../data/dataset/animalTrainedModel1.csv -l 1 -w 5 -h 5
//    ./trainGeneralImageNetwork.out -t RGB -d /../data/mnist_png/mnist_png/smallTesting -a /../data/mnist_png/mnist_png/indexToAnswerFile.txt -o /../data/mnist_png/mnist_png/mnistTrainedModel1.csv -l 1 -w 5 -h 5


#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <map>
#include <sys/resource.h>
#include <vector>

#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <coreutils/functions/debug/print.hpp>

#include <artificialIntelligence/basicLearningTypes/generationalAIBasic.cuh>
#include <artificialIntelligence/classes/layerLists/BasicLayerList.cuh>

#include <Image_Manipulation/generate/generateInput.hpp>

using namespace std;
using namespace imageEdit;
using namespace coreutils::classes::matrixes;
using namespace coreutils::functions;

#define MAXCOUNT -1

void loadImages (std::string inputImageFolder, Matrix3D **inputMatrixes, Matrix3D **outputMatrixes, map<std::string, int>& answerToIndex, std::string type, int size, int imageWidth, int imageLength);

void loadImagesHelper (std::string inputImageFolder, Matrix3D **inputMatrixes, Matrix3D **outputMatrixes, map<std::string, int>& answerToIndex, std::string type, int size, int* index, int imageWidth, int imageLength);

int countImages(std::string inputImageFolder, std::string indexToAnswerPath, map<std::string, int>& answerToIndex);

int countImagesHelper (std::string inputImageFolder, ofstream &indexToAnswerFile, map<std::string, int>& answerToIndex, int* index, int depth = 0);

void help (char *argv[]) {
   fprintf(stderr, "Usage: %s [-e epochs] [-r learningRate] [-l length] [-w width] [-h height] [-c layer count] [-b batch size] [-t type of image (BW, RGB, RGBA)] [-d path to data (required)] [-a index to answer file (required)] [-o model output file (required)]\n",argv[0]);
   exit(EXIT_FAILURE);
}

int main (int argc, char *argv[]) {
   int startTime = time(0);
   std::cout << std::fixed;
   std::cout.precision(2);


   // get program options
   int opt;

   int epochs = 15;
   double learningRate = 0.02;
   int hiddenLayerLength = 1; 
   int hiddenLayerWidth = 10;
   int hiddenLayerHeight = 100;
	int batchSize = 4;
   int imageLength = 240;
   int imageWidth = 320;

   // layerCount - 2 = hiddenLayerCount
   int layerCount = 5;

   std::string path = "";
   std::string indexToAnswerPath = "";
   std::string outputFile = "";
   std::string type = "BW";
   string* currentPath = new string (filesystem::current_path());
   *currentPath += "/../../../artificialIntelligence/neuralNetworks";
   filesystem::current_path(filesystem::path(*currentPath));

   while ((opt = getopt(argc, argv, "e:r:l:w:h:c:b:t:d:a:o:n:m:")) != -1) {
      switch (opt) {
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
         case 't':
            type = optarg;
            if (type != "BW" && type != "RGB" && type != "RGBA") {
               help(argv);
            }
         case 'd':
            path = optarg;
            break;
         case 'a':
            indexToAnswerPath = optarg;
            break;
         case 'o':
            outputFile = optarg;
            break;
         case 'n':
            imageLength = atoi(optarg);
            break;
         case 'm':
            imageWidth = atoi(optarg);
            break;
         default: /* '?' */ 
            help(argv);
         }
   }
   if (path == "" || indexToAnswerPath == "" || outputFile == "") {
      help(argv);
   }

   string inputImageFolder = (string) filesystem::current_path() + path;
   indexToAnswerPath = (string) filesystem::current_path() + indexToAnswerPath;
   outputFile = (string) filesystem::current_path() + outputFile;
   delete currentPath;

   map<std::string, int> answerToIndex;
   int inputCount = countImages (inputImageFolder, indexToAnswerPath, answerToIndex);

   std::cout << "ImageCount " << inputCount << '\n';
   
   Matrix3D** inputMatrixes = new Matrix3D* [inputCount];
   Matrix3D** outputMatrixes = new Matrix3D* [inputCount];

   loadImages (inputImageFolder, inputMatrixes, outputMatrixes, answerToIndex, type, inputCount, imageWidth, imageLength);

   std::cout << "100.00 percent of the images have been loaded\n";

	std::cout << "Generating Model Layer 1 of " << layerCount << ".\n";
   BasicLayerList* model = new artificialIntelligence::classes::BasicLayerList ();

   // input layer
   model->add (inputMatrixes[0]);
   model->copyRootMatrix(inputMatrixes[0]);

   // hidden layers
   for (int i = 0; i < layerCount - 2; i++) {
		std::cout << "Generating Model Layer " << i + 2 << " of " << layerCount << ".\n";
      model->addNew (hiddenLayerLength, hiddenLayerWidth, hiddenLayerHeight);
   }

	std::cout << "Generating Model Layer " << layerCount << " of " << layerCount << ".\n";
   // this is the output layer
   model->addNew (outputMatrixes[0]->getLength(), outputMatrixes[0]->getWidth(), outputMatrixes[0]->getHeight());

	std::cout << "Model Generated.\n\n";
   // model->getRoot()->getLayerMatrix()->printMatrix();
   // model->getLast()->getLayerMatrix()->printMatrix();
   // std::cout << "epochs: " << epochs << '\n';
   // std::cout << "learningRate: " << learningRate << '\n';
   // std::cout << "inputCount: " << inputCount << '\n';
   // model->print(false, false);
	// exit(0);
	// model->print(1,1);

   if (MAXCOUNT >= 1 && MAXCOUNT < inputCount) {
      inputCount = MAXCOUNT;
   }

   outputMatrixes[0]->printMatrix();
   outputMatrixes[500]->printMatrix();
   outputMatrixes[1000]->printMatrix();
   outputMatrixes[3000]->printMatrix();
   outputMatrixes[5000]->printMatrix();

   outputMatrixes[0]->printMatrixSize();

   artificialIntelligence::basicLearningTypes::generationalAIBasic::runStochasticGradientDescent(model, epochs, learningRate, inputMatrixes, outputMatrixes, inputCount, batchSize, false, false);
	// model->print(1,1);
   model->toFile (outputFile);

   int final = time(0) - startTime;
   std::cout.precision(9);
   std::cout << "\nTime to Complete: " << std::fixed << final << "s\n";
   struct rusage usage;
   getrusage (RUSAGE_SELF, &usage);
   std::cout << "\nMemory used (MB): " << usage.ru_maxrss / 1000000 << "\n\n";
}

void loadImages (std::string inputImageFolder, Matrix3D **inputMatrixes, Matrix3D **outputMatrixes, map<std::string, int>& answerToIndex, std::string type, int size, int imageWidth, int imageLength) {
   int* index = new int(0);
   for (const auto & entry : filesystem::directory_iterator(inputImageFolder)) {
      std::cout << "name: " << entry.path() << " \n"; 
      loadImagesHelper (entry.path(), inputMatrixes, outputMatrixes, answerToIndex, type, size, index, imageWidth, imageLength);
   }
   std::cout << "\nThe images have finished loading. \n\n\n";
}

void loadImagesHelper (std::string inputImageFolder, Matrix3D **inputMatrixes, Matrix3D **outputMatrixes, map<std::string, int>& answerToIndex, std::string type, int size, int* index, int imageWidth, int imageLength) {
   


   //***TESTING***///
   if (*index == MAXCOUNT) {
      return;
   }



   if (inputImageFolder.find(".DS_Store") != std::string::npos) {
      std::cout << inputImageFolder.find(".DS_Store");
      return;
   }

   // its a file
   if (!std::filesystem::is_directory(inputImageFolder)) {
      std::cout << "name: " << inputImageFolder << " "; 
      generate::inputMatrixNormalized(inputImageFolder, inputMatrixes, *index, type, imageWidth, imageLength);
      
      // make the output
      Matrix3D* output = new Matrix3D (1, 1, answerToIndex.size());
      int last = inputImageFolder.find_last_of("/");
      int secondLast = inputImageFolder.rfind("/", last - 1);
      std::string name = inputImageFolder.substr(secondLast + 1, last - secondLast - 1);

      for (auto it = answerToIndex.begin(); it != answerToIndex.end(); ++it) {
         if (inputImageFolder.find(it->first) != std::string::npos) {
            output->insert(1, 0, 0, answerToIndex.at(it->first));
            break;
         }
      }

      outputMatrixes[*index] = output;
      (*index)++;
      std::cout << *index << "/" << size << " loaded\n";
      return;
   }

   for (const auto & entry : filesystem::directory_iterator(inputImageFolder)) {
      loadImagesHelper (entry.path(), inputMatrixes, outputMatrixes, answerToIndex, type, size, index, imageWidth, imageLength);
   }
}

int countImages(std::string inputImageFolder, std::string indexToAnswerPath, map<std::string, int>& answerToIndex) {
   ofstream indexToAnswerFile (indexToAnswerPath);
   std::cout << indexToAnswerPath << "\n";
   if (!indexToAnswerFile.good()){
      std::cout << "Index to answer file path failure.\n";
      exit (EXIT_FAILURE);
   }
   int index = 0;
   return countImagesHelper(inputImageFolder, indexToAnswerFile, answerToIndex, &index);
}

int countImagesHelper (std::string inputImageFolder, ofstream &indexToAnswerFile, map<std::string, int>& answerToIndex, int* index, int depth) {

   // its a file
   if (!std::filesystem::is_directory(inputImageFolder)) {
      return 1;
   }

   // its a dir
   int count = 0;
   std::vector<std::filesystem::path> files_in_directory;
   std::copy(std::filesystem::directory_iterator(inputImageFolder), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
   for (const std::string & entry : files_in_directory) {
      if (entry.find(".DS_Store") == std::string::npos) {
         count += countImagesHelper(entry, indexToAnswerFile, answerToIndex, index, depth + 1);

         // if it got a file in the next folder, use this directory name as the name for the answer file
         if (depth == 0) {
            if (entry.find_last_of("/") != std::string::npos) {
               std::string name = entry.substr(entry.find_last_of("/") + 1, entry.length() - entry.find_last_of("/") - 1);
               std::cout << *index << " " << name << "\n";
               indexToAnswerFile << *index << " " << name << "\n";
               answerToIndex.insert(pair<std::string, int>(name, *index));
               (*index)++;
            }
         }
      }
   }

   return count;

}