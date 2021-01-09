#include "../include/Particles.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include<stdio.h>
#include<time.h>

__global__ void _Particles_Initialize_createPrng(int seed, curandState* d_prngStates)
{
	int particleId = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, particleId, 0, &d_prngStates[particleId]);
}

Particles::Particles(Options* options) : options(options)
{
	particlesNumber = options->particlesNumber;
	dimensions = options->dimesions;

	cudaMalloc(&d_coordinates, particlesNumber * dimensions * sizeof(float));
	cudaMalloc(&d_prngStates, particlesNumber * dimensions * sizeof(curandState));

	_Particles_Initialize_createPrng << <options->getGridSizeInitialization(), options->getBlockSizeInitialization() >> >
		((int)time(NULL), d_prngStates);
}

Particles::~Particles()
{
	cudaFree(d_coordinates);
	cudaFree(d_prngStates);
}

void Particles::print()
{
	float* coordinates = new float[particlesNumber * dimensions * sizeof(float)];
	cudaMemcpy(coordinates, d_coordinates, particlesNumber * dimensions * sizeof(float),
		cudaMemcpyDeviceToHost);

	for (int particleId = 0; particleId < particlesNumber; particleId++)
	{
		printf("[%d] = (", particleId);
		int firstCoord = particleId * dimensions;
		for (int coordinate = 0; coordinate < dimensions; coordinate++)
		{
			if (coordinate != dimensions - 1)
				printf("% .2f,\t", coordinates[firstCoord + coordinate]);
			else
				printf("% .2f)\n", coordinates[firstCoord + coordinate]);
		}
	}
	delete coordinates;
}