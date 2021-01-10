#pragma once

#include "Options.cuh"
#include "PsoParticles.cuh"

class Pso
{
public:
	Pso(Options* options, PsoParticles* particles);
	virtual ~Pso() = default;

	void solve();

private:
	Options* options;
	PsoParticles* particles;
};