#include "../include/Particles.cuh"

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