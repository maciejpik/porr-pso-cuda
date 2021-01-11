#include "include/Options.cuh"

#include <cuda_runtime.h>

__constant__ int d_particlesNumber;
__constant__ int d_dimensions;
__constant__ boxConstraints d_initializationBoxConstraints;
__constant__ boxConstraints d_boxConstraints;
__constant__ psoConstants d_psoConstants;
__constant__ mcConstants d_mcConstants;

int main(int argc, char* argv[])
{
	Options* options = new Options(argc, argv);
	options->verbose = false;
}