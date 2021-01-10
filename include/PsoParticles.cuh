#pragma once

#include "Particles.cuh"
#include "Options.cuh"

#include <curand_kernel.h>

__global__ void _PsoParticles_Initialize_createParticles(float* d_coordinates, float* d_velocities,
	curandState* d_prngStates);
__global__ void _PsoParticles_updateLBest(float* d_coordinates, float* d_cost, float* d_lBestCoordinates,
	float* d_lBestCost);
__global__ void _PsoParticles_updatePosition(float* d_coordinates, float* d_velocities, float* d_gBestCoordinates,
	float* d_lBestCoordinates, curandState* d_prngStates);

class PsoParticles : public Particles
{
public:
	PsoParticles(Options* options);
	virtual ~PsoParticles();

	void updateGBest();
	void updateLBest();
	void updatePosition();

	float* getBestCoordinates();
	float* getBestCost();

protected:
	float* d_velocities;
	float* d_gBestCoordinates;
	float* d_gBestCost;
	float* d_lBestCoordinates;
	float* d_lBestCost;

	float* gBestCoordinates;
	float* gBestCost;
};