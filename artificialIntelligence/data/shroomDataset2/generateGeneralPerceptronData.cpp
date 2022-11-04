// creates an answer file, and a file for determining which number correlates to which name
// g++ generateGeneralPerceptronData.cpp -o generateGeneralPerceptronData.out -I ../../../../ -I ../../../ --std=c++2a

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

using namespace std;
using namespace coreutils::classes::matrixes;

int main (int argc, char **argv) {
   std::cout << std::fixed;
   std::cout << std::setprecision(2);

   string* currentPath = new string (filesystem::current_path());

   if (argc < 1) {
      std::cout << "Index of output layer required\n";
      // std::cout << "Print perceptron levels with comma seperated values, with a - representing the output node.\n";
      // std::cout << "Example: 3 types of plant, 2 shapes of plant, and 2 possible colors would be 2,3,-2\n";
      exit (0);
   }

   string dataPath = *currentPath + "/data.csv";
   ifstream* dataFile = new ifstream (dataPath);
   std::ofstream outputFile;
   outputFile.open (*currentPath + "/shroom2Key.csv");

   // int index = atoi(argv[1]);
   int rows = 0;
   int columns = 0;
   
   // get the first line and see how many columns there are
   std::string line;
   std::stringstream lineStream;
   std::string value;
   std::map<int, std::string> order;

   getline (*dataFile, line);
   lineStream << line;
   while (getline(lineStream, value, ',')) {
      order.insert (std::pair<int, std::string> (columns, value));
      std::cout << value << ',';
      columns++;
   }
   columns++;
   std::cout << "\n";

   std::map<std::string, std::vector<std::string>* >* map = new std::map<std::string, std::vector<std::string>* > ();
   lineStream.clear(); 
   lineStream.seekg(0, std::ios::beg);

   while (getline(lineStream, value, ',')) {
      map->insert (std::pair<std::string,std::vector<std::string>* >(value, new std::vector<std::string> ()));
   }

   // add the values to the vector if it isnt in the vector yet
   while (getline (*dataFile, line)) {
      int i = 0;
      std::stringstream lineStream;
      lineStream << line;
      while (getline(lineStream, value, ',')) {
         vector <std::string>* v = map->at (order.at(i));
         bool exists = false;
         for (int k = 0, cc = v->size(); k < cc; k++) {
            if (v->at(k) == value) {
               exists = true;
               k = cc;
            }
         }
         if (!exists) {
            v->push_back(value);
         }
         i++;
      }

      rows++;
   }

   std::cout << "columns: " << columns << '\n';
   std::cout << "rows: " << rows << '\n';
   for (auto it = order.begin(); it != order.end(); ++it) {
      vector <std::string>* v = map->at (it->second);
      outputFile << it->second << "\n";
      for (int k = 0; k < v->size() - 1; k++) {
         outputFile << v->at(k) << ",";
      }
      outputFile << v->at(v->size() - 1);
      outputFile << "\n";
   }

   dataFile->clear(); 
   dataFile->seekg(0, std::ios::beg);

   // std::cout << map->at (order.at (index))->size () << '\n';
   // Matrix3D** inputMatrixes = new Matrix3D* [rows];
   // Matrix3D** outputMatrixes = new Matrix3D* [map->at (order.at (index))->size ()];
   
}

