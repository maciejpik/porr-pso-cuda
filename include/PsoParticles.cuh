#pragma once

#include "Particles.cuh"
#include "Options.h"

#include <curand_kernel.h>

__global__ void _PsoParticles_Initialize_createParticles(float* d_coordinates, float* d_velocities,
	curandState* d_prngStates);

class PsoParticles : public Particles
{
public:
	PsoParticles(Options* options);
	virtual ~PsoParticles() = default;

	void updateGBest();

protected:
	float* d_velocities;
	float* d_gBestCoordinates;
	float* d_gBestCost;

	float* gBestCoordinates;
	float* gBestCost;
};