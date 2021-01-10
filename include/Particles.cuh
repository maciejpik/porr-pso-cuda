#pragma once

#include "Options.cuh"

#include <curand_kernel.h>

__global__ void _Particles_Initialize_createPrng(int seed, curandState* d_prngStates);
__global__ void _Particles_computeCost_Task1(float* d_coordinates, float* d_cost);
__global__ void _Particles_computeCost_Task2(float* d_coordinates, float* d_cost);
__device__ float _Particles_computeCost_Task1(float coordinateValue);
__device__ float _Particles_computeCost_Task2(float coordinateValue, float coordinateValuePlusOne);

class Particles
{
public:
	Particles(Options* options);
	virtual ~Particles();
	
	void print();
	void computeCost();

protected:
	int particlesNumber;
	int dimensions;

	Options* options;
	float* d_coordinates;
	float* d_cost;
	curandState* d_prngStates;
};