#include "../include/PsoParticles.cuh"
#include "../include/Particles.cuh"
#include "../include/Options.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <stdio.h>

const int maxDimensions = 128;

extern __constant__ int d_particlesNumber;
extern __constant__ int d_dimensions;
extern __constant__ boxConstraints d_initializationBoxConstraints;
extern __constant__ boxConstraints d_boxConstraints;
extern __constant__ psoConstants d_psoConstants;

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

__global__ void _PsoParticles_updateLBest(float* d_coordinates, float* d_cost, float* d_lBestCoordinates,
	float* d_lBestCost)
{
	int particleId = blockIdx.x;
	int dimension = threadIdx.x;

	if (d_lBestCost[particleId] < d_cost[particleId])
	{
		d_lBestCost[particleId] < d_cost[particleId];
		d_lBestCoordinates[particleId * d_dimensions + dimension] = d_coordinates[particleId * d_dimensions + dimension];
	}
}

__global__ void _PsoParticles_updatePosition(float* d_coordinates, float* d_velocities, float* d_gBestCoordinates,
	float* d_lBestCoordinates, curandState* d_prngStates)
{
	int particleId = blockIdx.x;
	int dimension = threadIdx.x;
	int globalId = threadIdx.x + blockDim.x * blockIdx.x;
	curandState prngLocalState = d_prngStates[particleId];

	float randLocal = curand_uniform(&prngLocalState);
	float randGlobal = curand_uniform(&prngLocalState);
	d_prngStates[particleId] = prngLocalState;

	float newVelocity;
	float newCoordinates;
	newVelocity = d_psoConstants.w * d_velocities[globalId] +
		d_psoConstants.speedLocal * randLocal * (d_lBestCoordinates[globalId] - d_coordinates[globalId]) +
		d_psoConstants.speedGlobal * randGlobal * (d_gBestCoordinates[dimension] - d_coordinates[globalId]);
	newCoordinates = d_coordinates[globalId] + newVelocity;
	
	__shared__ float k[maxDimensions];
	k[dimension] = -1;

	if (newCoordinates > d_boxConstraints.max)
		k[dimension] = (-newCoordinates + d_boxConstraints.max) / newVelocity;
	else if (newCoordinates < d_boxConstraints.min)
		k[dimension] = (-newCoordinates + d_boxConstraints.min) / newVelocity;
	__syncthreads();

	for (int i = maxDimensions >> 1; i > 0; i >>= 1)
	{
		if (dimension < i && dimension + i < d_dimensions)
		{
			if (k[dimension] < 0 && k[dimension + i] > 0)
				k[dimension] = k[dimension + i];
			else if (k[dimension] > k[dimension + i] && k[dimension + i] > 0)
				k[dimension] = k[dimension + i];
		}
		__syncthreads();
	}

	d_velocities[globalId] = k[0] * newVelocity;
	d_coordinates[globalId] += k[0] * newVelocity;
}

PsoParticles::PsoParticles(Options* options) : Particles(options)
{
	cudaMalloc(&d_velocities, particlesNumber * dimensions * sizeof(float));
	cudaMalloc(&d_gBestCoordinates, dimensions * sizeof(float));
	cudaMalloc(&d_gBestCost, sizeof(float));
	cudaMalloc(&d_lBestCoordinates, particlesNumber * dimensions * sizeof(float));
	cudaMalloc(&d_lBestCost, particlesNumber * sizeof(float));

	cudaMallocHost(&gBestCoordinates, dimensions * sizeof(float));
	cudaMallocHost(&gBestCost, sizeof(float));

	_PsoParticles_Initialize_createParticles << <options->getGridSizeInitialization(), options->getBlockSizeInitialization() >> >
		(d_coordinates, d_velocities, d_prngStates);
	computeCost();

	cudaMemcpy(gBestCost, d_cost, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(gBestCoordinates, d_coordinates, dimensions * sizeof(float),
		cudaMemcpyDeviceToHost);
	cudaMemcpy(d_gBestCost, d_cost, sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_gBestCoordinates, d_coordinates, dimensions * sizeof(float),
		cudaMemcpyDeviceToDevice);
	updateGBest();

	cudaMemcpy(d_lBestCost, d_cost, particlesNumber * sizeof(float),
		cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_lBestCoordinates, d_coordinates, particlesNumber * dimensions * sizeof(float),
		cudaMemcpyDeviceToDevice);
	updateLBest();
}

PsoParticles::~PsoParticles()
{
	cudaFree(d_velocities);
	cudaFree(d_gBestCoordinates);
	cudaFree(d_gBestCost);
	cudaFree(d_lBestCoordinates);
	cudaFree(d_lBestCost);

	cudaFreeHost(gBestCoordinates);
	cudaFreeHost(gBestCost);
}

void PsoParticles::updateGBest()
{
	computeCost();

	thrust::device_ptr<float> temp_d_cost(d_cost);
	thrust::device_ptr<float> temp_gBestCost = thrust::min_element(temp_d_cost,
		temp_d_cost + particlesNumber);

	if (temp_gBestCost[0] < *gBestCost)
	{
		int bestParticleId = &temp_gBestCost[0] - &temp_d_cost[0];
		*gBestCost = temp_gBestCost[0];
		cudaMemcpy(gBestCoordinates, (d_coordinates + bestParticleId * dimensions),
			dimensions * sizeof(float), cudaMemcpyDeviceToHost);

		cudaMemcpy(d_gBestCost, (d_cost + bestParticleId), sizeof(float),
			cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_gBestCoordinates, (d_coordinates + bestParticleId * dimensions),
			dimensions * sizeof(float), cudaMemcpyDeviceToDevice);
	}
}

void PsoParticles::updateLBest()
{
	_PsoParticles_updateLBest << <particlesNumber, dimensions >> > (d_coordinates,
		d_cost, d_lBestCoordinates, d_lBestCost);
}

void PsoParticles::updatePosition()
{
	_PsoParticles_updatePosition << <particlesNumber, dimensions >> >
		(d_coordinates, d_velocities, d_gBestCoordinates, d_lBestCoordinates,
			d_prngStates);
}

float* PsoParticles::getBestCoordinates()
{
	return gBestCoordinates;
}

float* PsoParticles::getBestCost()
{
	return gBestCost;
}