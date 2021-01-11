#include "../include/Particles.cuh"
#include "../include/Options.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>
#include <stdio.h>

extern __constant__ int d_particlesNumber;
extern __constant__ int d_dimensions;
extern __constant__ boxConstraints d_initializationBoxConstraints;
extern __constant__ boxConstraints d_solutionBoxConstraints;

__global__ void _Particles_Particles_initPrng(int seed, curandState* d_prngStates)
{
	int particleId = threadIdx.x + blockIdx.x * blockDim.x;
	if (particleId >= d_particlesNumber)
		return;
	curand_init(seed, particleId, 0, &d_prngStates[particleId]);
}

__global__ void _PsoParticles_computeCosts_Task1(float* d_positions, float* d_costs)
{
	int particleId = threadIdx.x + blockIdx.x * blockDim.x;
	if (particleId >= d_particlesNumber)
		return;

	float subSum = 0, subProduct = 1;
	for (int coordIdx = particleId, i = 1; coordIdx < d_particlesNumber * d_dimensions;
		coordIdx += d_particlesNumber, i++)
	{
		float x_i = d_positions[coordIdx];
		subSum += x_i * x_i;
		subProduct *= cosf(x_i / i);
	}

	d_costs[particleId] = subSum / 40.0 + 1 - subProduct;
}

__device__ float _PsoParticles_computeCosts_Task1(float* position)
{
	float subSum = 0, subProduct = 1;
	for (int i = 0; i < d_dimensions; i++)
	{
		float x_i = position[i];
		subSum += x_i * x_i;
		subProduct *= cosf(x_i / i);
	}

	return subSum / 40.0 + 1 - subProduct;
}

__global__ void _PsoParticles_computeCosts_Task2(float* d_positions, float* d_costs)
{
	int particleId = threadIdx.x + blockIdx.x * blockDim.x;
	if (particleId >= d_particlesNumber)
		return;

	float subSum = 0;
	int coordIdx = particleId;
	float x_i = 0, x_i_1 = d_positions[coordIdx];
	for (coordIdx += d_particlesNumber; coordIdx < d_particlesNumber * d_dimensions;
		coordIdx += d_particlesNumber)
	{
		x_i = x_i_1;
		x_i_1 = d_positions[coordIdx];
		subSum += 100 * (x_i_1 - x_i * x_i) * (x_i_1 - x_i * x_i) +
			(1 - x_i) * (1 - x_i);
	}

	d_costs[particleId] = subSum;
}

__device__ float _PsoParticles_computeCosts_Task2(float* position)
{
	float subSum = 0;
	float x_i = 0, x_i_1 = position[0];
	for (int i = 1; i < d_dimensions; i++)
	{
		x_i = x_i_1;
		x_i_1 = position[i];
		subSum += 100 * (x_i_1 - x_i * x_i) * (x_i_1 - x_i * x_i) +
			(1 - x_i) * (1 - x_i);
	}

	return subSum;
}

Particles::Particles(Options* options)
	: options(options)
{
	cudaMalloc(&d_positions, options->particlesNumber * options->dimensions * sizeof(float));
	cudaMalloc(&d_costs, options->particlesNumber * sizeof(float));
	cudaMalloc(&d_prngStates, options->particlesNumber * sizeof(curandState));

	_Particles_Particles_initPrng<<<options->gridSize, options->blockSize>>>(time(NULL), d_prngStates);
}

Particles::~Particles()
{
	cudaFree(d_positions);
	cudaFree(d_positions);
	cudaFree(d_prngStates);
}

void Particles::print()
{
	float* positions = new float[options->particlesNumber * options->dimensions * sizeof(float)];
	cudaMemcpy(positions, d_positions, options->particlesNumber * options->dimensions * sizeof(float),
		cudaMemcpyDeviceToHost);

	float* costs = new float[options->particlesNumber * sizeof(float)];
	cudaMemcpy(costs, d_costs, options->particlesNumber * sizeof(float), cudaMemcpyDeviceToHost);

	for (int particleId = 0; particleId < options->particlesNumber; particleId++)
	{
		printf("[%d] = (", particleId);
		int coordIdx;
		for (coordIdx = particleId; coordIdx < options->particlesNumber * (options->dimensions - 1);
			coordIdx += options->particlesNumber)
		{
			printf("% .2f,\t", positions[coordIdx]);
		}
		printf("% .2f)", positions[coordIdx]);
		printf("\t f(x) =\t% .2f\n", costs[particleId]);
	}

	delete positions, costs;
}

void Particles::updateCosts()
{
	if (options->task == options->taskType::TASK_1)
		_PsoParticles_computeCosts_Task1 << <options->gridSize, options->blockSize >> > (d_positions, d_costs);
	else if (options->task == options->taskType::TASK_2)
		_PsoParticles_computeCosts_Task2 << <options->gridSize, options->blockSize >> > (d_positions, d_costs);
}