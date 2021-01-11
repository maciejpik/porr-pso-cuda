#pragma once

#include "Particles.cuh"
#include "Options.cuh"

__global__ void _PsoParticles_PsoParticles_initialize(float* d_positions, float* d_velocities,
	curandState* d_prngStates);

class PsoParticles : public Particles
{
public:
	PsoParticles(Options* options);
	~PsoParticles();

private:
	float* d_velocities;
};