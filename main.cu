#include "include/Options.h"
#include "include/PsoParticles.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include<stdio.h>

__constant__ int d_particlesNumber;
__constant__ int d_dimensions;
__constant__ boxConstraints d_initializationBoxConstraints;
__constant__ boxConstraints d_boxConstraints;

int main(int argc, char* argv[])
{
	Options* options = new Options(argc, argv);

	PsoParticles particles(options);
	particles.print();
}