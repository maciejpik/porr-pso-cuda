#include "../include/McParticles.cuh"
#include "../include/Options.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

const int maxDimensions = 128;

extern __constant__ int d_particlesNumber;
extern __constant__ int d_dimensions;
extern __constant__ boxConstraints d_initializationBoxConstraints;
extern __constant__ boxConstraints d_boxConstraints;
extern __constant__ psoConstants d_psoConstants;
extern __constant__ mcConstants d_mcConstants;

__global__ void _McParticles_Initialize_createParticles(float* d_coordinates, curandState* d_prngStates)
{
	int creatorId = threadIdx.x + blockIdx.x * blockDim.x;
	curandState prngLocalState = d_prngStates[creatorId];

	for (int offset = 0; offset < d_particlesNumber * d_dimensions; offset += d_particlesNumber)
	{
		d_coordinates[offset + creatorId] = d_initializationBoxConstraints.min +
			(d_initializationBoxConstraints.max - d_initializationBoxConstraints.min) * curand_uniform(&prngLocalState);
	}

	d_prngStates[creatorId] = prngLocalState;
}

template<int taskId>
__global__ void _McParticles_updatePosition(float* d_coordinates, float* d_cost, curandState* d_prngStates)
{
	int particleId = blockIdx.x;
	int dimension = threadIdx.x;
	int globalId = threadIdx.x + blockDim.x * blockIdx.x;
	curandState prngLocalState = d_prngStates[particleId];

	float deltaCoordinates[maxDimensions];
	for (int i = 0; i < d_dimensions; i++)
		deltaCoordinates[i] = (-1 + 2*curand_uniform(&prngLocalState)) * d_mcConstants.sigma;

	float newVelocity = deltaCoordinates[dimension];
	float coordinate = d_coordinates[globalId];
	float newCoordinate = coordinate + newVelocity;

	__shared__ float k[maxDimensions];
	k[dimension] = -1;

	if (newCoordinate > d_boxConstraints.max)
		k[dimension] = (-newCoordinate + d_boxConstraints.max) / newVelocity;
	else if (newCoordinate < d_boxConstraints.min)
		k[dimension] = (-newCoordinate + d_boxConstraints.min) / newVelocity;
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
	if (k[0] < 0)
		k[0] = 1;

	newCoordinate = coordinate + k[0] * newVelocity;
	float newCost;

	if (taskId == 1)
	{
		newCost = _Particles_computeCost_Task1(newCoordinate);
	}
	else if (taskId == 2)
	{
		__shared__ float newCoordinates[maxDimensions];
		newCoordinates[dimension] = newCoordinate;
		__syncthreads();

		newCost = _Particles_computeCost_Task2(newCoordinates[dimension], newCoordinates[dimension + 1]);
	}

	float currentCost = d_cost[particleId];
	if (newCost < currentCost)
	{
		d_cost[particleId] = newCost;
		d_coordinates[globalId] = newCoordinate;
	}
	else
	{
		float rand = curand_uniform(&prngLocalState);
		float threshold = __expf(-(newCost - currentCost) / d_mcConstants.T);
		if (rand < threshold)
		{
			d_cost[particleId] = newCost;
			d_coordinates[globalId] = newCoordinate;
		}
	}
	d_prngStates[particleId] = prngLocalState;
}

template __global__ void _McParticles_updatePosition<1>(float* d_coordinates, float* d_cost, curandState* d_prngStates);
template __global__ void _McParticles_updatePosition<2>(float* d_coordinates, float* d_cost, curandState* d_prngStates);

McParticles::McParticles(Options* options) : Particles(options)
{
	lastBestParticleId = 0;

	cudaMallocHost(&gBestCoordinates, dimensions * sizeof(float));
	cudaMallocHost(&gBestCost, sizeof(float));

	_McParticles_Initialize_createParticles << <options->getGridSizeInitialization(), options->getBlockSizeInitialization() >> >
		(d_coordinates, d_prngStates);
	computeCost();

	cudaMemcpy(gBestCost, d_cost, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(gBestCoordinates, d_coordinates, dimensions * sizeof(float),
		cudaMemcpyDeviceToHost);
}

McParticles::~McParticles()
{
	cudaFreeHost(gBestCoordinates);
	cudaFreeHost(gBestCost);
}

void McParticles::updatePosition()
{
	if(options->task == options->taskType::TASK_1)
		_McParticles_updatePosition <1> << <particlesNumber, dimensions >> > (d_coordinates, d_cost, d_prngStates);
	else if (options->task == options->taskType::TASK_2)
		_McParticles_updatePosition <2> << <particlesNumber, dimensions >> > (d_coordinates, d_cost, d_prngStates);
}

float* McParticles::getBestCoordinates()
{
	cudaMemcpy(gBestCoordinates, (d_coordinates + lastBestParticleId * dimensions),
		dimensions * sizeof(float), cudaMemcpyDeviceToHost);

	return gBestCoordinates;
}

float* McParticles::getBestCost()
{
	thrust::device_ptr<float> temp_d_cost(d_cost);
	thrust::device_ptr<float> temp_gBestCost = thrust::min_element(temp_d_cost,
		temp_d_cost + particlesNumber);

	if (temp_gBestCost[0] < *gBestCost)
	{
		int lastbestParticleId = &temp_gBestCost[0] - &temp_d_cost[0];
		*gBestCost = temp_gBestCost[0];
	}

	return gBestCost;
}