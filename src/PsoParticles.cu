#include "../include/PsoParticles.cuh"
#include "../include/Options.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

extern __constant__ int d_particlesNumber;
extern __constant__ int d_dimensions;
extern __constant__ boxConstraints d_initializationBoxConstraints;
extern __constant__ boxConstraints d_solutionBoxConstraints;
extern __constant__ psoConstants d_psoConstants;

__global__ void _PsoParticles_PsoParticles_initialize(float* d_positions, float* d_velocities,
	curandState* d_prngStates)
{
	int particleId = threadIdx.x + blockIdx.x * blockDim.x;
	if (particleId > d_particlesNumber)
		return;
	curandState prngLocalState = d_prngStates[particleId];
	
	for (int coordIdx = particleId; coordIdx < d_particlesNumber * d_dimensions;
		coordIdx += d_particlesNumber)
	{
		d_positions[coordIdx] = d_initializationBoxConstraints.min +
			(d_initializationBoxConstraints.max - d_initializationBoxConstraints.min) * curand_uniform(&prngLocalState);
		d_velocities[coordIdx] = (d_solutionBoxConstraints.min +
			(d_solutionBoxConstraints.max - d_solutionBoxConstraints.min) * curand_uniform(&prngLocalState))
			/ (d_solutionBoxConstraints.max - d_solutionBoxConstraints.min);
	}

	d_prngStates[particleId] = prngLocalState;
}

PsoParticles::PsoParticles(Options* options)
	: Particles(options)
{
	cudaMalloc(&d_velocities, options->particlesNumber * options->dimensions * sizeof(float));

	_PsoParticles_PsoParticles_initialize << <options->gridSize, options->blockSize >> > (d_positions,
		d_velocities, d_prngStates);
}

PsoParticles::~PsoParticles()
{
	cudaFree(d_velocities);
}