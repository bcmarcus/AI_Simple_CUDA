#include <stdio.h>
#include <coreutils/classes/matrixes/Matrix3D.cuh>
#include <artificialIntelligence/classes/layerLists/BasicLayerList.cuh>
#include <coreutils/util/cudaErrors.cuh>

using namespace coreutils::classes::matrixes;


__global__ void kernel(float* d1, float* d2, float* d3, int l, int w, int h) {
   for (int i = 0; i < l; i++) {
      for (int j = 0; j < w; j++) {
         for (int k = 0; k < h; k++) {
            int index = i * w * h + j * h + k;
            d3[index] = d1[index] + d2[index];
            printf("d1[%d][%d][%d] + d2[%d][%d][%d] = d3[%d][%d][%d]   ::::   %f + %f = %f\n", i, j, k, i, j, k, i, j, k, d1[index], d2[index], d3[index]);
         }
      }
   }
}



void addTwoMatrixes(Matrix3D *m1, Matrix3D *m2, Matrix3D *m3) {
   m1->randomize();
   m2->randomize();
   if (m1->getLength() != m2->getLength() || m1->getLength() != m3->getLength()) {
      std::cout << "length error\n";
   }
   if (m1->getWidth() != m2->getWidth() || m1->getWidth() != m3->getWidth()) {
      std::cout << "width error\n";
   }
   if (m1->getHeight() != m2->getHeight() || m1->getHeight() != m3->getHeight()) {
      std::cout << "height error\n";
   }

   int l = m1->getLength();
   int w = m1->getWidth();
   int h = m1->getHeight();

   for (int i = 0; i < l; i++) {
      for (int j = 0; j < w; j++) {
         for (int k = 0; k < h; k++) {
            m2->insert(1, i, j, k);
         }
      }
   }

   float* d1;
   float* d2;
   float* d3;
   
   gpuErrchk(cudaMalloc((void **) &d1, m1->getSize()));
   gpuErrchk(cudaMemcpy(d1, m1->getArr(), m1->getSize(), cudaMemcpyHostToDevice));

   gpuErrchk(cudaMalloc((void **) &d2, m2->getSize()));
   gpuErrchk(cudaMemcpy(d2, m2->getArr(), m2->getSize(), cudaMemcpyHostToDevice));

   gpuErrchk(cudaMalloc((void **) &d3, m3->getSize()));
   gpuErrchk(cudaMemcpy(d3, m3->getArr(), m3->getSize(), cudaMemcpyHostToDevice));
   
   kernel<<<1, 1>>>(d1, d2, d3, l, w, h);
   gpuErrchk(cudaDeviceSynchronize());

   gpuErrchk(cudaMemcpy(m3->getArr(), d3, m3->getSize(), cudaMemcpyDeviceToHost));

}

int main() {
   int l = 1, w = 2, h = 3;

   Matrix3D* m1 = new Matrix3D(l, w, h);
   Matrix3D* m2 = new Matrix3D(l, w, h);
   Matrix3D* m3 = new Matrix3D(l, w, h);

   addTwoMatrixes(m1, m2, m3);

	BasicLayerList* model = new BasicLayerList();

	Matrix3D* m4 = new Matrix3D(3, 28, 28);
	m1->printMatrix();
	model->add(m4);
	model->add(m2);
	model->add(m3);
}