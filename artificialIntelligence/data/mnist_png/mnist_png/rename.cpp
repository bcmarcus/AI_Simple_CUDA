#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <string>

using namespace std;

int main() {

   string* currentPath = new string (filesystem::current_path());

   std::string inputImageFolder1 = (string) filesystem::current_path() + "/training/";
   std::string inputImageFolder2 = (string) filesystem::current_path() + "/testing/";
   // std::string inputImageFolder3 = (string) filesystem::current_path() + "/smallSet/";

   std::string map[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
   for (int i = 0; i < 10; i++) {
      std::cout << rename ((inputImageFolder1 + std::to_string(i)).c_str(), (inputImageFolder1 + map[i]).c_str());
      int counter = 0;
      for (const auto & entry : filesystem::directory_iterator(inputImageFolder1 + map[i])) {
         std::cout << rename (entry.path().c_str(), (inputImageFolder1 + map[i] + "/" + map[i] + "_" + std::to_string(counter++) + ".png").c_str());
      }
   }
   for (int i = 0; i < 10; i++) {
      std::cout << rename ((inputImageFolder2 + std::to_string(i)).c_str(), (inputImageFolder2 + map[i]).c_str());
      int counter = 0;
      for (const auto & entry : filesystem::directory_iterator(inputImageFolder2 + map[i])) {
         std::cout << rename (entry.path().c_str(), (inputImageFolder2 + map[i] + "/" + map[i] + "_" + std::to_string(counter++) + ".png").c_str());
      }
   }
}