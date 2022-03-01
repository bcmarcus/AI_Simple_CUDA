#include <artificialIntelligence/classes/Basic3DWeightList.hpp>
#include <artificialIntelligence/classes/BasicWeight.hpp>
#include <artificialIntelligence/classes/BasicLayerList.hpp>
#include <coreutils/classes/matrixes/Matrix3D.cpp>

using namespace artificialIntelligence::classes;

// makes a Basic3DWeightList with new, random weights at each spot with matrix size of sl, sw, sh

Basic3DWeightList::Basic3DWeightList (int fl, int fw, int fh, int sl, int sw, int sh) {
   this->root = new BasicWeight(fl, fw, fh, sl, sw, sh);
}

// makes an empty Basic3DWeightList

Basic3DWeightList::Basic3DWeightList () {
   this->root = nullptr;
}


void Basic3DWeightList::print() {
   this->root->print();
}

// gets the weights for the specific node

Matrix3D* Basic3DWeightList::getWeightMatrix (int length, int width, int height) {
   return this->root->getWeightMatrix (length, width, height);
}

// puts weights in a specific spot. They can not be put in without a spot

void Basic3DWeightList::insert (int length, int width, int height, Matrix3D weights) {
   this->root = &(this->root->add (length, width, height, &weights));
}