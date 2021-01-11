#pragma once

#include "Particles.cuh"
#include "Options.cuh"

#include <curand_kernel.h>

__global__ void _PsoParticles_PsoParticles_initialize(float* d_positions, float* d_velocities,
	curandState* d_prngStates);
__global__ void _PsoParticles_updateLBest(float* d_positions, float* d_costs, float* d_lBestPositions,
	float* d_lBestCosts);
__global__ void _PsoParticles_updatePositions(float* d_positions, float* d_velocities, float* d_gBestPosition,
	float* d_lBestPositions, curandState* d_prngStates);

class PsoParticles : public Particles
{
public:
	PsoParticles(Options* options);
	~PsoParticles();

	void updateGBest();
	void updateLBest();
	void updatePositions();

private:
	float* d_velocities;
	float* d_gBestPosition;
	float* d_gBestCost;
	float* d_lBestPositions;
	float* d_lBestCosts;

	float* gBestPosition;
	float* gBestCost;
};