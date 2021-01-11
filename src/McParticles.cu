#include "../include/McParticles.cuh"
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
extern __constant__ mcConstants d_mcConstants;

__global__ void _McParticles_McParticles_initialize(float* d_positions, curandState* d_prngStates)
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
	}

	d_prngStates[particleId] = prngLocalState;
}

template<int taskId, int maxDimension>
__global__ void _McParticles_McParticles_initialize(float* d_positions, float* d_costs,
	curandState* d_prngStates)
{
	int particleId = threadIdx.x + blockIdx.x * blockDim.x;
	if (particleId >= d_particlesNumber)
		return;
	curandState prngLocalState = d_prngStates[particleId];

	float deltaPosition[maxDimension];
	float newPosition[maxDimension];

	float k = 1;
	for (int coordIdx = particleId, i = 0; coordIdx < d_particlesNumber * d_dimensions;
		coordIdx += d_particlesNumber, i++)
	{
		deltaPosition[i] = (-1 + 2 * curand_uniform(&prngLocalState)) * d_mcConstants.sigma;
		newPosition[i] = d_positions[coordIdx] + deltaPosition[i];

		float k_temp = 1;
		if (newPosition[i] > d_solutionBoxConstraints.max)
			k_temp = (-newPosition[i] + d_solutionBoxConstraints.max) / deltaPosition[i];
		else if (newPosition[i] < d_solutionBoxConstraints.min)
			k_temp = (-newPosition[i] + d_solutionBoxConstraints.min) / deltaPosition[i];

		if (k_temp < k && k_temp > 0)
			k = k_temp;
	}

	for (int coordIdx = particleId, i = 0; coordIdx < d_particlesNumber * d_dimensions;
		coordIdx += d_particlesNumber, i++)
		newPosition[i] += d_positions[coordIdx] + k * deltaPosition[i];

	float newCost;
	if(taskId == 1)
		newCost = _PsoParticles_computeCosts_Task1(newPosition);
	else if(taskId == 2)
		newCost = _PsoParticles_computeCosts_Task2(newPosition);

	float currentCost = d_costs[particleId];
	if (newCost < currentCost)
	{
		d_costs[particleId] = newCost;
		for (int coordIdx = particleId, i = 0; coordIdx < d_particlesNumber * d_dimensions;
			coordIdx += d_particlesNumber, i++)
			d_positions[coordIdx] = newPosition[i];
	}
	else
	{
		float rand = curand_uniform(&prngLocalState);
		float threshold = __expf(-(newCost - currentCost) / d_mcConstants.T);
		if (rand < threshold)
		{
			d_costs[particleId] = newCost;
			for (int coordIdx = particleId, i = 0; coordIdx < d_particlesNumber * d_dimensions;
				coordIdx += d_particlesNumber, i++)
				d_positions[coordIdx] = newPosition[i];
		}
	}

	d_prngStates[particleId] = prngLocalState;
}

template __global__ void _McParticles_McParticles_initialize<1, 16>(float* d_positions, float* d_costs,
	curandState* d_prngStates);
template __global__ void _McParticles_McParticles_initialize<2, 16>(float* d_positions, float* d_costs,
	curandState* d_prngStates);
template __global__ void _McParticles_McParticles_initialize<1, 32>(float* d_positions, float* d_costs,
	curandState* d_prngStates);
template __global__ void _McParticles_McParticles_initialize<2, 32>(float* d_positions, float* d_costs,
	curandState* d_prngStates);
template __global__ void _McParticles_McParticles_initialize<1, 64>(float* d_positions, float* d_costs,
	curandState* d_prngStates);
template __global__ void _McParticles_McParticles_initialize<2, 64>(float* d_positions, float* d_costs,
	curandState* d_prngStates);
template __global__ void _McParticles_McParticles_initialize<1, 128>(float* d_positions, float* d_costs,
	curandState* d_prngStates);
template __global__ void _McParticles_McParticles_initialize<2, 128>(float* d_positions, float* d_costs,
	curandState* d_prngStates);
template __global__ void _McParticles_McParticles_initialize<1, 256>(float* d_positions, float* d_costs,
	curandState* d_prngStates);
template __global__ void _McParticles_McParticles_initialize<2, 256>(float* d_positions, float* d_costs,
	curandState* d_prngStates);
template __global__ void _McParticles_McParticles_initialize<1, 512>(float* d_positions, float* d_costs,
	curandState* d_prngStates);
template __global__ void _McParticles_McParticles_initialize<2, 512>(float* d_positions, float* d_costs,
	curandState* d_prngStates);


McParticles::McParticles(Options* options)
	: Particles(options)
{
	bestParticleId = 0;

	cudaMallocHost(&gBestPosition, options->dimensions * sizeof(float));
	cudaMallocHost(&gBestCost, sizeof(float));

	_McParticles_McParticles_initialize << <options->gridSize, options->blockSize >> >
		(d_positions, d_prngStates);
	updateCosts();

	cudaMemcpy2D(gBestPosition, sizeof(float), d_positions, options->particlesNumber * sizeof(float),
		sizeof(float), options->dimensions, cudaMemcpyDeviceToHost);
	cudaMemcpy(gBestCost, d_costs, sizeof(float), cudaMemcpyDeviceToHost);
}

McParticles::~McParticles()
{
	cudaFreeHost(gBestPosition);
	cudaFreeHost(gBestCost);
}

float* McParticles::getBestPosition()
{
	cudaMemcpy2D(gBestPosition, sizeof(float), d_positions + bestParticleId,
		options->particlesNumber * sizeof(float),
		sizeof(float), options->dimensions, cudaMemcpyDeviceToHost);

	return gBestPosition;
}

float* McParticles::getBestCost()
{
	thrust::device_ptr<float> temp_d_costs(d_costs);
	thrust::device_ptr<float> temp_gBestCost = thrust::min_element(temp_d_costs,
		temp_d_costs + options->particlesNumber);

	if (temp_gBestCost[0] < *gBestCost)
	{
		bestParticleId = &temp_gBestCost[0] - &temp_d_costs[0];
		*gBestCost = temp_gBestCost[0];
	}

	return gBestCost;
}

void McParticles::updatePosition()
{
	if (options->dimensions < 16)
	{
		if (options->task == options->taskType::TASK_1)
			_McParticles_McParticles_initialize<1, 16> << <options->gridSize, options->blockSize >> >
				(d_positions, d_costs, d_prngStates);
		else if (options->task == options->taskType::TASK_2)
			_McParticles_McParticles_initialize<2, 16> << <options->gridSize, options->blockSize >> >
				(d_positions, d_costs, d_prngStates);
	}
	else if (options->dimensions < 32)
	{
		if (options->task == options->taskType::TASK_1)
			_McParticles_McParticles_initialize<1, 32> << <options->gridSize, options->blockSize >> >
			(d_positions, d_costs, d_prngStates);
		else if (options->task == options->taskType::TASK_2)
			_McParticles_McParticles_initialize<2, 32> << <options->gridSize, options->blockSize >> >
			(d_positions, d_costs, d_prngStates);
	}
	else if (options->dimensions < 64)
	{
		if (options->task == options->taskType::TASK_1)
			_McParticles_McParticles_initialize<1, 64> << <options->gridSize, options->blockSize >> >
			(d_positions, d_costs, d_prngStates);
		else if (options->task == options->taskType::TASK_2)
			_McParticles_McParticles_initialize<2, 64> << <options->gridSize, options->blockSize >> >
			(d_positions, d_costs, d_prngStates);
	}
	else if (options->dimensions < 128)
	{
		if (options->task == options->taskType::TASK_1)
			_McParticles_McParticles_initialize<1, 128> << <options->gridSize, options->blockSize >> >
			(d_positions, d_costs, d_prngStates);
		else if (options->task == options->taskType::TASK_2)
			_McParticles_McParticles_initialize<2, 128> << <options->gridSize, options->blockSize >> >
			(d_positions, d_costs, d_prngStates);
	}
	else if(options->dimensions < 256)
	{
		if (options->task == options->taskType::TASK_1)
			_McParticles_McParticles_initialize<1, 256> << <options->gridSize, options->blockSize >> >
			(d_positions, d_costs, d_prngStates);
		else if (options->task == options->taskType::TASK_2)
			_McParticles_McParticles_initialize<2, 256> << <options->gridSize, options->blockSize >> >
			(d_positions, d_costs, d_prngStates);
	}
	else
	{
		if (options->task == options->taskType::TASK_1)
			_McParticles_McParticles_initialize<1, 512> << <options->gridSize, options->blockSize >> >
			(d_positions, d_costs, d_prngStates);
		else if (options->task == options->taskType::TASK_2)
			_McParticles_McParticles_initialize<2, 512> << <options->gridSize, options->blockSize >> >
			(d_positions, d_costs, d_prngStates);
	}
}