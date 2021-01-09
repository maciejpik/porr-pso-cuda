#pragma once

#include "Options.h"

#include <curand_kernel.h>

__global__ void _Particles_Initialize_createPrng(int seed, curandState* d_prngStates);

class Particles
{
public:
	Particles(Options* options);
	virtual ~Particles();
	
	void print();

protected:
	int particlesNumber;
	int dimensions;

	Options* options;
	float* d_coordinates;
	float* d_cost;
	curandState* d_prngStates;
};