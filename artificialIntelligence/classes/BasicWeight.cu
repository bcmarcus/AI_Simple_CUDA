#include <iostream>
#include <string>

#include <coreutils/classes/matrixes/Matrix3D.cpp>

#include <artificialIntelligence/classes/BasicWeight.hpp>

using namespace coreutils::classes::matrixes;
using namespace artificialIntelligence::classes;


BasicWeight::BasicWeight (Matrix3D* weights){
   this->weights = weights;
   this->right = nullptr;
   this->back = nullptr;
   this->down = nullptr;
}


BasicWeight::BasicWeight (int fl, int fw, int fh, int sl, int sw, int sh) {
   this->weights = nullptr;
   this->right = nullptr;
   this->back = nullptr;
   this->down = nullptr;
   try {
      if (fl < 1 || fw < 1 || fh < 1 || sl < 1 || sw < 1 || sh < 1) {
         std::cout << " \nfl :: " << fl << " fw :: " << fw << " fh :: " << fh 
                  << " sl :: " << sl << " sw :: " << sw << " sh :: " << sh;
         throw std::invalid_argument("\nBasicWeight generate values are invalid\n\n");
      }
   } catch(const std::invalid_argument& e) {
      std::cout << e.what();
      exit (0);
   };
   this->generate (fl, fw, fh, sl, sw, sh);
   // std::cout << "\n\n" << std::to_string(this->generate (fl, fw, fh, sl, sw, sh) + 1) << "\n\n";
}


BasicWeight::BasicWeight (){
   this->weights = nullptr;
   this->right = nullptr;
   this->back = nullptr;
   this->down = nullptr;
}


BasicWeight::~BasicWeight(){
   // needs to traverse through all of the down, and then all of the back, and then all of the right, without repeating
   if (this->weights != nullptr) {
      delete this->weights;
   } else {
      return;
   }

   if (this->right != nullptr) {
      delete this->right;
   } 
   if (this->back != nullptr) {
      delete this->back;
   } 
   if (this->down != nullptr){
      delete this->down;
   }
}


void BasicWeight::print () {
   int total = this->print (0,0,0);
}


int BasicWeight::print (int length, int width, int height) {
   // needs to traverse through all of the down, and then all of the back, and then all of the right, without repeating
   if (this->weights != nullptr) {
      std::cout << "Weight Location: [" << length << "][" << width << "][" << height << "]\n";
      this->weights->printMatrix(); 
   } else {
      return 1;
      std::cout << "No weights found!\n";
   }

   int depth = 0;
   if (this->right != nullptr && width == 0 && height == 0) {
      depth += this->right->print (length + 1, width, height) + 1;
   } 
   if (this->back != nullptr && height == 0) {
      depth += this->back->print (length, width + 1, height) + 1;
   } 

   if (this->down != nullptr){
      depth += this->down->print (length, width, height + 1) + 1;
   }
   return depth;
}


BasicWeight* BasicWeight::add (int length, int width, int height, Matrix3D* weights) {
   if (length == 0) {
      if (width == 0) {
         if (height == 0){
            this->weights = weights;
            return this;
         } else {
            this->down = this->down->add (length, width, height - 1, weights);
            return this;
         }
      } else {
         this->back = this->back->add (length, width - 1, height, weights);
         return this;
      }
   } else {
      this->right = this->right->add (length - 1, width, height, weights);
      return this;
   }
}


BasicWeight* BasicWeight::addNew (int length, int width, int height) {
   Matrix3D* layer = new Matrix3D (length, width, height);
   layer->randomize ();
   return this->add (length, width, height, weights);
}

// broken

int BasicWeight::generate (int fl, int fw, int fh, int sl, int sw, int sh) {
   int i = 0;
   // std::cout << fl << " " << fw << " " << fh << " " << sl << " " << sw << " " << sh << '\n';
      // exit (0);
   Matrix3D* weights = new Matrix3D (sl, sw, sh);
   weights->randomize();
   
   this->weights = weights;
   if (fl > 1 && fw >= 1 && fh >= 1) {
      this->right = new BasicWeight();
      i += this->right->generate (fl - 1, fw, fh, sl, sw, sh) + 1;
   }
   if (fw > 1 && fh >= 1) {
      this->back = new BasicWeight();
      i += this->back->generate (1, fw - 1, fh, sl, sw, sh) + 1;
   }
   if (fh > 1) {
      this->down = new BasicWeight();
      i += this->down->generate (1, 1, fh - 1, sl, sw, sh) + 1;
   }
   return i;
}


Matrix3D* BasicWeight::getWeightMatrix (int length, int width, int height) {
   if (length == 0) {
      if (width == 0) {
         if (height == 0){
            return this->getWeightMatrix ();
         } else {
            if (this->down == nullptr) {
               return nullptr;
            }
            return this->down->getWeightMatrix (length, width, height - 1);
         }
      } else {
         if (this->back == nullptr) {
            return nullptr;
         }
         return this->back->getWeightMatrix (length, width - 1, height);
      }
   } else {
      if (this->right == nullptr) {
         return nullptr;
      }
      return this->right->getWeightMatrix (length - 1, width, height);
   }
}


Matrix3D* BasicWeight::getWeightMatrix () {
   return this->weights;
}


float* BasicWeight::getData (int fl, int fw, int fh, int sl, int sw, int sh) {
   Matrix3D* weights = this->getWeightMatrix(fl, fw, fh);
   if (weights == nullptr) {
      return nullptr;
   }
   return weights->getData(sl, sw, sh);
}


void BasicWeight::insert (float data, int fl, int fw, int fh, int sl, int sw, int sh) {
   Matrix3D* weights = this->getWeightMatrix(fl, fw, fh);
   if (weights == nullptr) {
      return;
   }
   weights->insert(data, sl, sw, sh);
}