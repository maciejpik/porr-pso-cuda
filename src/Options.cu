#include "../include/Options.h"

#include <cuda_runtime.h>

extern __constant__ int d_particlesNumber;
extern __constant__ int d_dimensions;
extern __constant__ boxConstraints d_initializationBoxConstraints;
extern __constant__ boxConstraints d_boxConstraints;

Options::Options()
{
	particlesNumber = 10;
	dimesions = 3;

	initializationBoxConstraints = { -40, 40 };
	boxConstraints = { -40, 40 };

	cudaMemcpyToSymbol(&d_particlesNumber, &particlesNumber, sizeof(int));
	cudaMemcpyToSymbol(&d_dimensions, &dimesions, sizeof(int));

	cudaMemcpyToSymbol(&d_initializationBoxConstraints, &initializationBoxConstraints,
		sizeof(boxConstraints));
	cudaMemcpyToSymbol(&d_boxConstraints, &boxConstraints,
		sizeof(boxConstraints));

	setBlockSizeInitialization(64);
}

int Options::getBlockSizeInitialization()
{
	return blockSize_initialization;
}

int Options::getGridSizeInitialization()
{
	return gridSize_initialization;
}

void Options::setBlockSizeInitialization( int blockSize )
{
	blockSize_initialization = blockSize;
	gridSize_initialization = (particlesNumber + blockSize - 1) / blockSize;
}