﻿#include "../include/Pso.cuh"
#include "../include/Options.cuh"
#include "../include/PsoParticles.cuh"

#include <stdio.h>
#include <chrono>

Pso::Pso(Options* options, PsoParticles* particles)
	: options(options), particles(particles) {}

void Pso::solve()
{
	auto tStart = std::chrono::high_resolution_clock::now();

	int iteration = 0, stop = 0;
	float cost = *(particles->getBestCost());

	if (options->verbose)
		printf("Iteration %4d: cost = %.4f\n", iteration, cost);

	FILE* logFile;
	char filename[100];
	if (options->logger)
	{
		sprintf(filename, "logFile_CUDA_par_%d_dim_%d_blockSize_%d_PSO_Task%d.txt",
			options->particlesNumber, options->dimensions, options->blockSize,
			(options->task + 1));
		logFile = fopen(filename, "w");
		fprintf(logFile, "%d\t%.10f\n", iteration, cost);
	}

	while (!stop )
	{
		iteration++;

		particles->updatePositions();
		particles->updateCosts();
		particles->updateGBest();
		particles->updateLBest();
	
		cost = *(particles->getBestCost());

		if (options->verbose)
			printf("Iteration %4d: cost = %.4f\n", iteration, cost);

		if (options->logger)
			fprintf(logFile, "%d\t%.10f\n", iteration, cost);

		if (cost < options->stopCriterion)
			stop = true;
	}
	auto tEnd = std::chrono::high_resolution_clock::now();
	long long int duration = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();

	printf("Solution f(x) = %.8f found after %d iterations (%lf s)\n", cost, iteration, duration / 1000000.0);

	if (options->logger)
		fclose(logFile);
}