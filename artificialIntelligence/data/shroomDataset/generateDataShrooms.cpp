// creates an answer file, and a file for determining which number correlates to which name
// g++ generateDataShrooms.cpp -o generateDataShrooms.out -I ../../../../ -I ../../../ --std=c++2a

#include <sys/time.h>
#include <sys/resource.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>

#include <Image_Manipulation/generate/generateInput.hpp>

#include <coreutils/functions/sort/sortHelpers.hpp>
#include <coreutils/functions/sort/sortingAlgorithms.hpp>

using namespace std;
using namespace imageEdit;
using namespace coreutils::classes::matrixes;

int main (int argc, char **argv) {
   std::cout << std::fixed;
   std::cout << std::setprecision(2);

   string* currentPath = new string (filesystem::current_path());

   ofstream mapFile (*currentPath + "/numberShroomMap.txt");
   ofstream answerFile (*currentPath + "/shroomAnswerFile.txt");
//    ifstream translationFile (*currentPath + "/translation.json");
//    string json;
   string dataSetFolder = *currentPath + "/dataset";
//    getline(translationFile, json);

   int i = 0;

   //copy all of the files into "files_in_directory" and sort them 
   std::vector<std::filesystem::path> files_in_directory;
   std::copy(std::filesystem::directory_iterator(dataSetFolder), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
   std::sort(files_in_directory.begin(), files_in_directory.end());

   // go through the sorted "files_in_directory"
   for (const std::string & folderName : files_in_directory) {
      std::vector<std::filesystem::path> files_in_sub_directory;
      std::copy(std::filesystem::directory_iterator(folderName), std::filesystem::directory_iterator(), std::back_inserter(files_in_sub_directory));
      std::sort(files_in_sub_directory.begin(), files_in_sub_directory.end());
      for (const std::string & fileName : files_in_sub_directory) {
         // go into the first directory
         filesystem::directory_entry folder (fileName);
         string path = folder.path();
         std::vector<std::filesystem::path> imgs;
         std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(imgs));

         //***KEEP***///
         // renaming files
         //
         // int j = 0;
         // for (const std::string & img : imgs) {
         //    std::cout << img << '\n';
         //    std::cout << (path + "/" + std::to_string(j) + img.substr(img.find_last_of('.'))) << '\n';
         //    std::cout << rename (img.c_str(), (path + "/" + std::to_string(j) + img.substr(img.find_last_of('.'))).c_str());
         //    j++;
         // }
         //***KEEP***/// 
         for (int k = 0; k < std::distance(filesystem::directory_iterator(path), filesystem::directory_iterator{}); k++) {
            answerFile << i << '\n';
         }
      }
      mapFile << i << " " << folderName.substr(std::strlen(dataSetFolder.c_str()) + 1) << '\n';
      i++;
   }
   
}