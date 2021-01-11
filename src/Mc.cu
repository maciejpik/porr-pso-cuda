﻿#include "../include/Mc.cuh"
#include "../include/Options.cuh"
#include "../include/McParticles.cuh"

#include <stdio.h>
#include <chrono>

Mc::Mc(Options* options, McParticles* particles)
	: options(options), particles(particles) {}

void Mc::solve()
{
	auto tStart = std::chrono::high_resolution_clock::now();

	int iteration = 0, stop = 0;
	float cost = *(particles->getBestCost());

	if (options->verbose)
		printf("Iteration %4d: cost = %.4f\n", iteration, cost);

	while (!stop)
	{
		iteration++;

		particles->updatePositions();

		cost = *(particles->getBestCost());

		if (options->verbose)
			printf("Iteration %4d: cost = %.4f\n", iteration, cost);

		if (cost < options->stopCriterion)
			stop = true;
	}
	auto tEnd = std::chrono::high_resolution_clock::now();
	long long int duration = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();

	printf("Solution f(x) = %.8f found after %d iterations (%lf s)\n", cost, iteration, duration / 1000000.0);
}