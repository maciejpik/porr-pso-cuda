#pragma once

#include "Particles.cuh"
#include "Options.cuh"

#include <curand_kernel.h>

__global__ void _McParticles_Initialize_createParticles(float* d_coordinates, curandState* d_prngStates);
template<int> __global__ void _McParticles_updatePosition(float* d_coordinates, float* d_cost, curandState* d_prngStates);

class McParticles : public Particles
{
public:
	McParticles(Options* options);
	virtual ~McParticles();

	void updatePosition();

	float* getBestCoordinates();
	float* getBestCost();

protected:
	float* gBestCoordinates;
	float* gBestCost;
	int lastBestParticleId;
};