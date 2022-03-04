#ifndef MATRIX3D_HPP
#define MATRIX3D_HPP

namespace coreutils {
   namespace classes {
      namespace matrixes {
         class Matrix3D{
            public:

               int getLength ();

               int getWidth ();

               int getHeight ();

               float* getArr ();

               long long getSize ();

               float* getData (int length, int width, int height);

               int getIndex (int l, int w, int h);

               // shuffles every single value
               void shuffleEvery ();

               // shuffles every 2d matrix. while retaining the 2d matrix
               int* shuffleGroups ();
               void shuffleGroups (int* order);

               // adds this and another matrix and 
               // sets this matrix equal to it
               void operator += (const Matrix3D* m2);

               // subtracts this and another matrix and 
               // sets this matrix equal to it 
               void operator -= (const Matrix3D* m2);

               // returns addition of this and another matrix
               Matrix3D* operator + (const Matrix3D* m2);

               // returns subtraction of this and another matrix
               Matrix3D* operator - (const Matrix3D* m2);

               // returns multiplication of this and another matrix
               Matrix3D* operator * (const Matrix3D* m2);
					
					// returns this matrix scaled by a value
               Matrix3D* operator * (const float x);

					// returns division of this and another matrix
               Matrix3D* operator / (const Matrix3D* m2);

					bool equals (const Matrix3D* m2);

               void randomize (double lowerBound = -0.5, double upperBound = 0.5);

               double dotProduct (const Matrix3D* m2);

               double sum ();

               void insert (float data, int length, int width, int height);

               void printMatrix ();

               void setMatrix (Matrix3D* M3D);

					void setAll (double x);

               Matrix3D (const int length, const int width, const int height);
					Matrix3D (const Matrix3D& m3d);
               Matrix3D ();

               ~Matrix3D ();
					
				private:
               // length is first, then width, then height
               int length;
               int width;
               int height;
               long long memorySize;
               float* arr;

         };
      }
   }
}

#endif