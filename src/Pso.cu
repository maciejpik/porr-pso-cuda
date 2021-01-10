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

	if(options->verbose)
		printf("Iteration %4d: cost = %.4f\n", iteration, cost);
	while (!stop)
	{
		iteration++;

		particles->updatePosition();
		particles->updateGBest();
		particles->updateLBest();

		cost = *(particles->getBestCost());

		if(options->verbose)
			printf("Iteration %4d: cost = %.4f\n", iteration, cost);

		if (cost < options->stopCriterion)
			stop = true;
	}
	auto tEnd = std::chrono::high_resolution_clock::now();
	long long int duration = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();

	printf("Solution f(x) = %.4f found after %d iterations (%lld ms)\n", cost, iteration, duration);
}