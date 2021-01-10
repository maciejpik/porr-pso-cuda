#include "../include/Particles.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include<stdio.h>
#include<time.h>
#include<math.h>

const int maxDimensions = 128;

extern __constant__ int d_particlesNumber;
extern __constant__ int d_dimensions;
extern __constant__ boxConstraints d_initializationBoxConstraints;
extern __constant__ boxConstraints d_boxConstraints;

__global__ void _Particles_Initialize_createPrng(int seed, curandState* d_prngStates)
{
	int particleId = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, particleId, 0, &d_prngStates[particleId]);
}

__global__ void _Particles_computeCost_Task1(float* d_coordinates, float* d_cost)
{
	__shared__ float subSum[maxDimensions];
	__shared__ float subMult[maxDimensions];

	int globalCoordinateId = threadIdx.x + blockIdx.x * blockDim.x;
	float coordinateValue = d_coordinates[globalCoordinateId];
	
	subSum[threadIdx.x] = coordinateValue * coordinateValue;
	subMult[threadIdx.x] = cosf(coordinateValue / (threadIdx.x + 1));
	__syncthreads();

	for (int i = maxDimensions >> 1; i > 0; i >>= 1)
	{
		if (threadIdx.x < i && threadIdx.x + i < blockDim.x)
		{
			subSum[threadIdx.x] += subSum[threadIdx.x + i];
			subMult[threadIdx.x] *= subMult[threadIdx.x + i];
		}
		__syncthreads();
	}
	
	d_cost[blockIdx.x] = subSum[0] / 40.0 + 1 - subMult[0];
	
}
__global__ void _Particles_computeCost_Task2(float* d_coordinates, float* d_cost)
{
	__shared__ float subSum[maxDimensions];
	__shared__ float localCoordinates[maxDimensions];
	localCoordinates[threadIdx.x] = d_coordinates[threadIdx.x + blockIdx.x * blockDim.x];
	__syncthreads();

	subSum[threadIdx.x] = 100 * (localCoordinates[threadIdx.x + 1] - localCoordinates[threadIdx.x] * localCoordinates[threadIdx.x]) *
		(localCoordinates[threadIdx.x + 1] - localCoordinates[threadIdx.x] * localCoordinates[threadIdx.x]) +
		(1 - localCoordinates[threadIdx.x] * localCoordinates[threadIdx.x]) *
		(1 - localCoordinates[threadIdx.x] * localCoordinates[threadIdx.x]);
	__syncthreads();

	for (int i = maxDimensions >> 1; i > 0; i >>= 1)
	{
		if (threadIdx.x < i && threadIdx.x + i < blockDim.x - 1)
			subSum[threadIdx.x] += subSum[threadIdx.x + i];
		__syncthreads();
	}

	d_cost[blockIdx.x] = subSum[0];
}

Particles::Particles(Options* options) : options(options)
{
	particlesNumber = options->particlesNumber;
	dimensions = options->dimesions;

	cudaMalloc(&d_coordinates, particlesNumber * dimensions * sizeof(float));
	cudaMalloc(&d_prngStates, particlesNumber * dimensions * sizeof(curandState));
	cudaMalloc(&d_cost, particlesNumber * sizeof(float));

	_Particles_Initialize_createPrng << <options->getGridSizeInitialization(), options->getBlockSizeInitialization() >> >
		((int)time(NULL), d_prngStates);
}

Particles::~Particles()
{
	cudaFree(d_coordinates);
	cudaFree(d_cost);
	cudaFree(d_prngStates);
}

void Particles::print()
{
	float* coordinates = new float[particlesNumber * dimensions * sizeof(float)];
	cudaMemcpy(coordinates, d_coordinates, particlesNumber * dimensions * sizeof(float),
		cudaMemcpyDeviceToHost);

	float* cost = new float[particlesNumber * sizeof(float)];
	cudaMemcpy(cost, d_cost, particlesNumber * sizeof(float), cudaMemcpyDeviceToHost);

	for (int particleId = 0; particleId < particlesNumber; particleId++)
	{
		printf("[%d] = (", particleId);
		int firstCoord = particleId * dimensions;
		for (int coordinate = 0; coordinate < dimensions; coordinate++)
		{
			if (coordinate != dimensions - 1)
				printf("% .2f,\t", coordinates[firstCoord + coordinate]);
			else
				printf("% .2f)", coordinates[firstCoord + coordinate]);
		}
		printf("\t f(x) =\t% .2f\n", cost[particleId]);
	}
	delete coordinates;
}

void Particles::computeCost()
{
	if (options->task == options->taskType::TASK_1)
	{
		_Particles_computeCost_Task1 << <particlesNumber, dimensions >> > (d_coordinates, d_cost);
	}
	else if (options->task == options->taskType::TASK_2)
		_Particles_computeCost_Task2 << <particlesNumber, dimensions >> > (d_coordinates, d_cost);
}