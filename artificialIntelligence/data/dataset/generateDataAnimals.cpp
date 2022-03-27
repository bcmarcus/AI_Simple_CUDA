// creates an answer file, and a file for determining which number correlates to which name
// g++ generateDataAnimals.cpp -o generateDataAnimals.out -I ../../../../ -I ../../../ --std=c++2a

#include <sys/time.h>
#include <sys/resource.h>
#include <filesystem>
#include <fstream>

#include <imageEdit/generate/generateInput.hpp>

#include <coreutils/functions/sort/sortHelpers.cpp>
#include <coreutils/functions/sort/sortingAlgorithms.cpp>

using namespace std;
using namespace imageEdit;
using namespace coreutils::classes::matrixes;

int main (int argc, char **argv) {
   std::cout << std::fixed;
   std::cout << std::setprecision(2);

   string* currentPath = new string (filesystem::current_path());

   ofstream mapFile (*currentPath + "/numberAnimalMap.txt");
   ofstream answerFile (*currentPath + "/animalAnswerFile.txt");
   ifstream translationFile (*currentPath + "/translation.json");
   string json;
   string dataSetFolder = *currentPath + "/dataset";
   getline(translationFile, json);


   // this needs to be used in the actual testing file
   // std::vector<std::filesystem::path> files_in_directory;
   // std::copy(std::filesystem::directory_iterator(inputImageFolder), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
   // std::sort(files_in_directory.begin(), files_in_directory.end());

   // for (const std::string & filename : files_in_directory) {
   //    filesystem::directory_entry entry (filename);
   // }


   int i = 0;

   std::vector<std::filesystem::path> files_in_directory;
   std::copy(std::filesystem::directory_iterator(dataSetFolder), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
   std::sort(files_in_directory.begin(), files_in_directory.end());

   for (const std::string & folderName : files_in_directory) {
      filesystem::directory_entry folder (folderName);
      string path = folder.path();
      string scientificName = path.substr(path.find("dataset/dataset/") + 16);
      if (scientificName != "." && scientificName != ".." && scientificName != ".DS_Store") {
         std::cout << scientificName << " ";
         int index = json.find(scientificName) + scientificName.length() + 4;
         string normalName = json.substr (index, json.find ("\"", index) - index);
         std::cout << normalName << '\n';
         mapFile << i << " " << normalName << '\n';
         std::cout << path << '\n';
         for (int k = 0; k < std::distance(filesystem::directory_iterator(path), filesystem::directory_iterator{}); k++) {
            answerFile << i << '\n';
         }
         i++;
      }
   }
   
}