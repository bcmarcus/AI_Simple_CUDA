#ifndef CUDA_ERRORS_CUH
#define CUDA_ERRORS_CUH

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

#endif