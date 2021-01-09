#include "../include/PsoParticles.cuh"
#include "../include/Particles.cuh"
#include "../include/Options.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <stdio.h>

extern __constant__ int d_particlesNumber;
extern __constant__ int d_dimensions;
extern __constant__ boxConstraints d_initializationBoxConstraints;
extern __constant__ boxConstraints d_boxConstraints;

__global__ void _PsoParticles_Initialize_createParticles(float* d_coordinates, float* d_velocities,
	curandState* d_prngStates)
{
	//int particleId = threadIdx.x + blockIdx.x * blockDim.x;
	//curandState prngLocalState = d_prngStates[particleId];

	//for (int particleCoord = 0; particleCoord < d_dimensions; particleCoord++)
	//{
	//	d_coordinates[particleId * d_dimensions + particleCoord] = d_initializationBoxConstraints.min +
	//		(d_initializationBoxConstraints.max - d_initializationBoxConstraints.min) * curand_uniform(&prngLocalState);
	//	d_velocities[particleId * d_dimensions + particleCoord] = (d_initializationBoxConstraints.min +
	//		(d_initializationBoxConstraints.max - d_initializationBoxConstraints.min) * curand_uniform(&prngLocalState))
	//		/ (d_initializationBoxConstraints.max - d_initializationBoxConstraints.min);
	//}

	//d_prngStates[particleId] = prngLocalState;
	int creatorId = threadIdx.x + blockIdx.x * blockDim.x;
	curandState prngLocalState = d_prngStates[creatorId];

	for (int offset = 0; offset < d_particlesNumber * d_dimensions; offset += d_particlesNumber)
	{
		d_coordinates[offset + creatorId] = d_initializationBoxConstraints.min +
			(d_initializationBoxConstraints.max - d_initializationBoxConstraints.min) * curand_uniform(&prngLocalState);
		d_velocities[offset + creatorId] = (d_initializationBoxConstraints.min +
			(d_initializationBoxConstraints.max - d_initializationBoxConstraints.min) * curand_uniform(&prngLocalState))
			/ (d_initializationBoxConstraints.max - d_initializationBoxConstraints.min);
	}

	d_prngStates[creatorId] = prngLocalState;
}

PsoParticles::PsoParticles(Options* options) : Particles(options)
{
	cudaMalloc(&d_velocities, particlesNumber * dimensions * sizeof(float));

	_PsoParticles_Initialize_createParticles << <options->getGridSizeInitialization(), options->getBlockSizeInitialization() >> >
		(d_coordinates, d_velocities, d_prngStates);
}