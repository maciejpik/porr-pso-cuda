#include "../include/PsoParticles.cuh"
#include "../include/Particles.cuh"
#include "../include/Options.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <stdio.h>

extern __constant__ int d_particlesNumber;
extern __constant__ int d_dimensions;
extern __constant__ boxConstraints d_initializationBoxConstraints;
extern __constant__ boxConstraints d_boxConstraints;

__global__ void _PsoParticles_Initialize_createParticles(float* d_coordinates, float* d_velocities,
	curandState* d_prngStates)
{
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
	cudaMalloc(&d_gBestCoordinates, dimensions * sizeof(float));
	cudaMalloc(&d_gBestCost, sizeof(float));

	cudaMallocHost(&gBestCoordinates, dimensions * sizeof(float));
	cudaMallocHost(&gBestCost, sizeof(float));

	_PsoParticles_Initialize_createParticles << <options->getGridSizeInitialization(), options->getBlockSizeInitialization() >> >
		(d_coordinates, d_velocities, d_prngStates);
	computeCost();
	updateGBest();
}

void PsoParticles::updateGBest()
{
	thrust::device_ptr<float> temp_d_cost(d_cost);
	thrust::device_ptr<float> temp_gBestCost = thrust::min_element(temp_d_cost,
		temp_d_cost + particlesNumber);

	int bestParticleId = &temp_gBestCost[0] - &temp_d_cost[0];
	*gBestCost = temp_gBestCost[0];
	cudaMemcpy(gBestCoordinates, (d_coordinates + bestParticleId*dimensions),
		dimensions * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(d_gBestCost, (d_cost + bestParticleId), sizeof(float),
		cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_gBestCoordinates, (d_coordinates + bestParticleId * dimensions),
		dimensions * sizeof(float), cudaMemcpyDeviceToDevice);
}