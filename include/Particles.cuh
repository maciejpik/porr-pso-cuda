#pragma once

#include "Options.cuh"

#include <curand_kernel.h>

__global__ void _Particles_Particles_initPrng(int seed, curandState* d_prngStates);

class Particles
{
public:
	Particles(Options* options);
	virtual ~Particles();

	void print();

protected:
	Options* options;
	float* d_positions;
	float* d_costs;
	curandState* d_prngStates;
};