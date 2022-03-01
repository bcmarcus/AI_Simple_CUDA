#ifndef SIMPLE_MATH_HPP
#define SIMPLE_MATH_HPP

namespace coreutils 
{
   namespace functions
   {
      namespace math {
         // determines if value is prime
         bool isPrime (int prime);

         // random number between beginning and end
         double rand (int beginning, int end);

         // determines if value is a palindrome (only works with ints)
         bool isPalindrome (int p);

         // counts the number of digits in a number
         int countDigits (int n);
		}
	}
}

#endif