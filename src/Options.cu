#include "../include/Options.cuh"

#include <cuda_runtime.h>

#include <stdio.h>

extern __constant__ int d_particlesNumber;
extern __constant__ int d_dimensions;
extern __constant__ boxConstraints d_initializationBoxConstraints;
extern __constant__ boxConstraints d_boxConstraints;
extern __constant__ psoConstants d_psoConstants;
extern __constant__ mcConstants d_mcConstants;

Options::Options(int argc, char* argv[])
{
	if (argc == 3)
	{
		sscanf(argv[1], "%d", &particlesNumber);
		sscanf(argv[2], "%d", &dimesions);
	}
	else
	{
		particlesNumber = 10;
		dimesions = 3;
	}

	initializationBoxConstraints = { -40, 40 };
	boxConstraints = { -40, 40 };
	float chi = 0.72984f, c1 = 2.05f, c2 = 2.05f;
	psoConstants = { chi, chi * c1, chi * c2 };
	mcConstants = { .1f, .01f };
	task = taskType::TASK_1;
	stopCriterion = 0.01f;
	verbose = true;

	cudaMemcpyToSymbol(&d_particlesNumber, &particlesNumber, sizeof(int));
	cudaMemcpyToSymbol(&d_dimensions, &dimesions, sizeof(int));
	cudaMemcpyToSymbol(&d_initializationBoxConstraints, &initializationBoxConstraints,
		sizeof(boxConstraints));
	cudaMemcpyToSymbol(&d_boxConstraints, &boxConstraints, sizeof(boxConstraints));
	cudaMemcpyToSymbol(&d_psoConstants, &psoConstants, sizeof(psoConstants));
	cudaMemcpyToSymbol(&d_mcConstants, &mcConstants, sizeof(mcConstants));
}