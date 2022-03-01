#include <sys/time.h>
#include <iostream>

struct timeval GetTimeStamp(){
	struct timeval tv;
   gettimeofday(&tv,__null);
   return tv;
}