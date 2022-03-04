//    ./trainGeneralImageNetwork -t RGB -d /../data/images/dataset/dataset -a /../data/images/dataset/animalAnswerFile.txt -o /../data/images/dataset/animalTrainedModel1.csv -h 5 -w 5
//    ./trainGeneralImageNetwork -t RGB -d /../data/mnist_png/mnist_png/smallTesting -a /../data/mnist_png/mnist_png/indexToAnswerFile.txt -o /../data/mnist_png/mnist_png/mnistTrainedModel1.csv -h 5 -w 5


#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <map>
#include <sys/resource.h>
#include <vector>

#include <artificialIntelligence/functions/images/generateInput.hpp>
#include <artificialIntelligence/basicLearningTypes/generationalAIBasic.hpp>
#include <artificialIntelligence/classes/BasicLayerList.hpp>
#include <coreutils/classes/matrixes/Matrix3D.cuh>

using namespace std;
using namespace artificialIntelligence::functions;
using namespace coreutils::classes::matrixes;

void loadImages (std::string inputImageFolder, Matrix3D **inputMatrixes, Matrix3D **outputMatrixes, map<std::string, int>& answerToIndex, std::string type, int size);

void loadImagesHelper (std::string inputImageFolder, Matrix3D **inputMatrixes, Matrix3D **outputMatrixes, map<std::string, int>& answerToIndex, std::string type, int size, int* index);

int countImages(std::string inputImageFolder, std::string indexToAnswerPath, map<std::string, int>& answerToIndex);

int countImagesHelper (std::string inputImageFolder, ofstream &indexToAnswerFile, map<std::string, int>& answerToIndex, int* index);

void help (char *argv[]) {
   fprintf(stderr, "Usage: %s [-e epochs] [-r learningRate] [-l length] [-w width] [-h height] [-c layer count] [-t type of image (BW, RGB, RGBA)] [-d path to data (required)] [-a index to answer file (required)] [-o model output file (required)]\n",argv[0]);
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

   // layerCount - 2 = hiddenLayerCount
   int layerCount = 5;

   std::string path = "";
   std::string indexToAnswerPath = "";
   std::string outputFile = "";
   std::string type = "BW";
   string* currentPath = new string (filesystem::current_path());
   *currentPath += "/../../../artificialIntelligence/neuralNetworks";
   filesystem::current_path(filesystem::path(*currentPath));

   while ((opt = getopt(argc, argv, "e:r:l:w:h:c:t:d:a:o:")) != -1) {
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

   loadImages (inputImageFolder, inputMatrixes, outputMatrixes, answerToIndex, type, inputCount);

   std::cout << "100.00 percent of the images have been loaded\n";

   BasicLayerList* model = new artificialIntelligence::classes::BasicLayerList ();
	
   // input layer
   model->add (inputMatrixes[0]);
   model->editRootMatrix(inputMatrixes[0]);

   // hidden layers
   for (int i = 0; i < layerCount - 2; i++) {
      model->addNew (hiddenLayerLength, hiddenLayerWidth, hiddenLayerHeight);
   }
   

	
   // this is the output layer
   model->addNew (outputMatrixes[0]->getLength(), outputMatrixes[0]->getWidth(), outputMatrixes[0]->getHeight());

   // model->getRoot()->getLayerMatrix()->printMatrix();
   // model->getLast()->getLayerMatrix()->printMatrix();
   // std::cout << "epochs: " << epochs << '\n';
   // std::cout << "learningRate: " << learningRate << '\n';
   // std::cout << "inputCount: " << inputCount << '\n';
   // model->print();

   artificialIntelligence::basicLearningTypes::generationalAIBasic::run(model, epochs, learningRate, inputMatrixes, outputMatrixes, inputCount);

   model->toFile (outputFile);

   int final = time(0) - startTime;
   std::cout.precision(9);
   std::cout << "\nTime to Complete: " << std::fixed << final << "s\n";
   struct rusage usage;
   getrusage (RUSAGE_SELF, &usage);
   std::cout << "\nMemory used (MB): " << usage.ru_maxrss / 1000000 << "\n\n";
}

void loadImages (std::string inputImageFolder, Matrix3D **inputMatrixes, Matrix3D **outputMatrixes, map<std::string, int>& answerToIndex, std::string type, int size) {
   int* index = new int(0);
   for (const auto & entry : filesystem::directory_iterator(inputImageFolder)) {
      std::cout << "name: " << entry.path() << " \n"; 
      loadImagesHelper (entry.path(), inputMatrixes, outputMatrixes, answerToIndex, type, size, index);
   }
   std::cout << "\nThe images have finished loading. \n\n\n";
}

void loadImagesHelper (std::string inputImageFolder, Matrix3D **inputMatrixes, Matrix3D **outputMatrixes, map<std::string, int>& answerToIndex, std::string type, int size, int* index) {
   //answerToIndex.at("tursiops-truncatus")
   if (inputImageFolder.find(".DS_Store") != std::string::npos) {
      std::cout << inputImageFolder.find(".DS_Store");
      return;
   }
   if (!std::filesystem::is_directory(inputImageFolder)) {
      std::cout << "name: " << inputImageFolder << " "; 
      images::generate::inputMatrixNormalized(inputImageFolder, inputMatrixes, *index, type);

      // make the output
      Matrix3D* output = new Matrix3D (1, 1, answerToIndex.size());
      int last = inputImageFolder.find_last_of("/");
      std::string name = inputImageFolder.substr(last + 1, inputImageFolder.find("_", (last + 1)) - last - 1);
      output->insert(1, 0, 0, answerToIndex.at(name));
      outputMatrixes[*index] = output;
      (*index)++;
      std::cout << *index << "/" << size << " loaded\n";
      return;
   }
   for (const auto & entry : filesystem::directory_iterator(inputImageFolder)) {
      loadImagesHelper (entry.path(), inputMatrixes, outputMatrixes, answerToIndex, type, size, index);
   }
}

int countImages(std::string inputImageFolder, std::string indexToAnswerPath, map<std::string, int>& answerToIndex) {
   ofstream indexToAnswerFile (indexToAnswerPath);
   std::cout << indexToAnswerPath;
   if (!indexToAnswerFile.good()){
      std::cout << "Index to answer file path failure.\n";
      exit (EXIT_FAILURE);
   }
   int* index = new int(0);
   std::cout << *index << '\n';
   return countImagesHelper(inputImageFolder, indexToAnswerFile, answerToIndex, index);
}

int countImagesHelper (std::string inputImageFolder, ofstream &indexToAnswerFile, map<std::string, int>& answerToIndex, int* index) {
   if (!std::filesystem::is_directory(inputImageFolder)) {
      return 1;
   }
   int count = 0;
   std::vector<std::filesystem::path> files_in_directory;
   std::copy(std::filesystem::directory_iterator(inputImageFolder), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));

   for (const std::string & entry : files_in_directory) {
      if (entry.find(".DS_Store") == std::string::npos) {
         count += countImagesHelper(entry, indexToAnswerFile, answerToIndex, index);

         // if it got a file in the next folder, use this directory name as the name for the answer file
         if (count == 1) {
            std::cout << "\n" << entry << "\n";
            if (inputImageFolder.find_last_of("/") != std::string::npos) {
               std::string name = inputImageFolder.substr(inputImageFolder.find_last_of("/") + 1, inputImageFolder.length() - inputImageFolder.find_last_of("/") - 1);
               std::cout<< *index << " " << name << "\n";
               indexToAnswerFile << *index << " " << name << "\n";
               answerToIndex.insert(pair<std::string, int>(name, *index));
               (*index)++;
            }
         }
      }
   }
   return count;

}