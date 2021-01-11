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
	if (particleId >= d_particlesNumber)
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
	if (particleId >= d_particlesNumber)
		return;

	if (d_costs[particleId] < d_lBestCosts[particleId])
	{
		d_lBestCosts[particleId] = d_costs[particleId];
		for (int coordIdx = particleId; coordIdx < d_particlesNumber * d_dimensions;
			coordIdx += d_particlesNumber)
			d_lBestPositions[coordIdx] = d_positions[coordIdx];
	}
}

template<int maxDimension>
__global__ void _PsoParticles_updatePositions(float* d_positions, float* d_velocities, float* d_gBestPosition,
	float* d_lBestPositions, curandState* d_prngStates)
{
	int particleId = threadIdx.x + blockIdx.x * blockDim.x;
	if (particleId >= d_particlesNumber)
		return;
	curandState prngLocalState = d_prngStates[particleId];

	__shared__ float gBestPosition[maxDimension];
	for (int i = threadIdx.x; i < d_dimensions; i += blockDim.x)
		gBestPosition[i] = d_gBestPosition[i];
	__syncthreads();

	float randLocal = curand_uniform(&prngLocalState);
	float randGlobal = curand_uniform(&prngLocalState);
	d_prngStates[particleId] = prngLocalState;

	float newVelocity[maxDimension];
	float newPosition[maxDimension];

	float k = 1;
	for (int coordIdx = particleId, i = 0; coordIdx < d_particlesNumber * d_dimensions;
		coordIdx += d_particlesNumber, i++)
	{
		newVelocity[i] = d_psoConstants.w * d_velocities[coordIdx] +
			d_psoConstants.speedLocal * randLocal * (d_lBestPositions[coordIdx] - d_positions[coordIdx]) +
			d_psoConstants.speedGlobal * randGlobal * (gBestPosition[i] - d_positions[coordIdx]);
		newPosition[i] = d_positions[coordIdx] + newVelocity[i];

		float k_temp = 1;
		if (newPosition[i] > d_solutionBoxConstraints.max)
			k_temp = (-newPosition[i] + d_solutionBoxConstraints.max) / newVelocity[i];
		else if (newPosition[i] < d_solutionBoxConstraints.min)
			k_temp = (-newPosition[i] + d_solutionBoxConstraints.min) / newVelocity[i];

		if (k_temp < k && k_temp > 0)
			k = k_temp;
	}

	for (int coordIdx = particleId, i = 0; coordIdx < d_particlesNumber * d_dimensions;
		coordIdx += d_particlesNumber, i++)
	{
		d_positions[coordIdx] += k * newVelocity[i];
		d_velocities[coordIdx] = k * newVelocity[i];
	}
}

template __global__ void _PsoParticles_updatePositions<16>(float* d_positions, float* d_velocities, float* d_gBestPosition,
	float* d_lBestPositions, curandState* d_prngStates);
template __global__ void _PsoParticles_updatePositions<32>(float* d_positions, float* d_velocities, float* d_gBestPosition,
	float* d_lBestPositions, curandState* d_prngStates);
template __global__ void _PsoParticles_updatePositions<64>(float* d_positions, float* d_velocities, float* d_gBestPosition,
	float* d_lBestPositions, curandState* d_prngStates);
template __global__ void _PsoParticles_updatePositions<128>(float* d_positions, float* d_velocities, float* d_gBestPosition,
	float* d_lBestPositions, curandState* d_prngStates);
template __global__ void _PsoParticles_updatePositions<256>(float* d_positions, float* d_velocities, float* d_gBestPosition,
	float* d_lBestPositions, curandState* d_prngStates);
template __global__ void _PsoParticles_updatePositions<512>(float* d_positions, float* d_velocities, float* d_gBestPosition,
	float* d_lBestPositions, curandState* d_prngStates);

PsoParticles::PsoParticles(Options* options)
	: Particles(options)
{
	cudaMalloc(&d_velocities, options->particlesNumber * options->dimensions * sizeof(float));
	cudaMalloc(&d_gBestPosition, options->dimensions * sizeof(float));
	cudaMalloc(&d_gBestCost, sizeof(float));
	cudaMalloc(&d_lBestPositions, options->particlesNumber * options->dimensions * sizeof(float));
	cudaMalloc(&d_lBestCosts, options->particlesNumber * sizeof(float));

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
	cudaFree(d_lBestPositions);
	cudaFree(d_lBestCosts);

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
	_PsoParticles_updateLBest << <options->gridSize, options->blockSize >> >
		(d_positions, d_costs, d_lBestPositions, d_lBestCosts);
}

void PsoParticles::updatePositions()
{
	int dimensions = options->dimensions;
	if(dimensions < 16)
		_PsoParticles_updatePositions <16> << <options->gridSize, options->blockSize >> >
			(d_positions, d_velocities, d_gBestPosition, d_lBestPositions, d_prngStates);
	else if(dimensions < 32)
		_PsoParticles_updatePositions <32> << <options->gridSize, options->blockSize >> >
			(d_positions, d_velocities, d_gBestPosition, d_lBestPositions, d_prngStates);
	else if(dimensions < 64)
		_PsoParticles_updatePositions <64> << <options->gridSize, options->blockSize >> >
			(d_positions, d_velocities, d_gBestPosition, d_lBestPositions, d_prngStates);
	else if(dimensions < 128)
		_PsoParticles_updatePositions <128> << <options->gridSize, options->blockSize >> >
			(d_positions, d_velocities, d_gBestPosition, d_lBestPositions, d_prngStates);
	else if(dimensions < 256)
		_PsoParticles_updatePositions <256> << <options->gridSize, options->blockSize >> >
			(d_positions, d_velocities, d_gBestPosition, d_lBestPositions, d_prngStates);
	else
		_PsoParticles_updatePositions <512> << <options->gridSize, options->blockSize >> >
			(d_positions, d_velocities, d_gBestPosition, d_lBestPositions, d_prngStates);
}