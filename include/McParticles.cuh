#pragma once

#include "Particles.cuh"
#include "Options.cuh"

#include <curand_kernel.h>

__global__ void _McParticles_McParticles_initialize(float* d_positions, curandState* d_prngStates);
template<int> __global__ void _McParticles_McParticles_initialize(float* d_positions, float* d_costs,
	curandState* d_prngStates);

class McParticles : public Particles
{
public:
	McParticles(Options* options);
	~McParticles();

	void updatePositions();

	float* getBestPosition();
	float* getBestCost();

private:
	float* gBestPosition;
	float* gBestCost;
	int bestParticleId;
};