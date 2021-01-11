#include "../include/PsoParticles.cuh"
#include "../include/Options.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

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

__global__ void _PsoParticles_updateLBest(float* d_positions, float* d_costs, float* d_lBestPositions,
	float* d_lBestCosts)
{
	int particleId = threadIdx.x + blockIdx.x * blockDim.x;
	if (particleId > d_particlesNumber)
		return;

	if (d_costs[particleId] < d_lBestCosts[particleId])
	{
		for (int coordIdx = particleId; coordIdx < d_particlesNumber * d_dimensions;
			coordIdx += d_particlesNumber)
			d_lBestCosts[coordIdx] += d_costs[coordIdx];
	}
}

PsoParticles::PsoParticles(Options* options)
	: Particles(options)
{
	cudaMalloc(&d_velocities, options->particlesNumber * options->dimensions * sizeof(float));
	cudaMalloc(&d_gBestPosition, options->dimensions * sizeof(float));
	cudaMalloc(&d_gBestCost, sizeof(float));

	cudaMallocHost(&gBestPosition, options->dimensions * sizeof(float));
	cudaMallocHost(&gBestCost, sizeof(float));

	_PsoParticles_PsoParticles_initialize << <options->gridSize, options->blockSize >> > (d_positions,
		d_velocities, d_prngStates);
	updateCosts();

	cudaMemcpy2D(d_gBestPosition, sizeof(float), d_positions, options->particlesNumber * sizeof(float),
		sizeof(float), options->dimensions, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_gBestCost, d_costs, sizeof(float), cudaMemcpyDeviceToDevice);

	cudaMemcpy2D(gBestPosition, sizeof(float), d_positions, options->particlesNumber * sizeof(float),
		sizeof(float), options->dimensions, cudaMemcpyDeviceToHost);
	cudaMemcpy(gBestCost, d_costs, sizeof(float), cudaMemcpyDeviceToHost);

	updateGBest();

	cudaMemcpy(d_lBestPositions, d_positions, options->particlesNumber * options->dimensions * sizeof(float),
		cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_lBestCosts, d_costs, options->particlesNumber * sizeof(float),
		cudaMemcpyDeviceToDevice);
}

PsoParticles::~PsoParticles()
{
	cudaFree(d_velocities);
	cudaFree(d_gBestPosition);
	cudaFree(d_gBestCost);

	cudaFreeHost(gBestPosition);
	cudaFreeHost(gBestCost);
}

void PsoParticles::updateGBest()
{
	thrust::device_ptr<float> temp_d_costs(d_costs);
	thrust::device_ptr<float> temp_gBestCost = thrust::min_element(temp_d_costs,
		temp_d_costs + options->particlesNumber);

	if (temp_gBestCost[0] < *gBestCost)
	{
		int bestParticleId = &temp_gBestCost[0] - &temp_d_costs[0];
		
		cudaMemcpy2D(d_gBestPosition, sizeof(float), d_positions + bestParticleId,
			options->particlesNumber * sizeof(float),
			sizeof(float), options->dimensions, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_gBestCost, d_costs + bestParticleId, sizeof(float), cudaMemcpyDeviceToDevice);

		cudaMemcpy2D(gBestPosition, sizeof(float), d_positions + bestParticleId,
			options->particlesNumber * sizeof(float),
			sizeof(float), options->dimensions, cudaMemcpyDeviceToHost);
		*gBestCost = temp_gBestCost[0];
	}
}

void PsoParticles::updateLBest()
{
	_PsoParticles_updateLBest << <options->gridSize, options->blockSize >> > (d_positions, d_costs, d_lBestPositions, d_lBestCosts);
}