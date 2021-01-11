#pragma once

#include "Options.cuh"

#include <curand_kernel.h>

__global__ void _Particles_Particles_initPrng(int seed, curandState* d_prngStates);
__global__ void _PsoParticles_computeCosts_Task1(float *d_positions, float *d_costs);
__global__ void _PsoParticles_computeCosts_Task2(float* d_positions, float* d_costs);

class Particles
{
public:
	Particles(Options* options);
	virtual ~Particles();

	void print();
	void updateCosts();

protected:
	Options* options;
	float* d_positions;
	float* d_costs;
	curandState* d_prngStates;
};