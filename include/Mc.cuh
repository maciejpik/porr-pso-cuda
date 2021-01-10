#pragma once

#include "Options.cuh"
#include "McParticles.cuh"

class Mc
{
public:
	Mc(Options* options, McParticles* particles);
	virtual ~Mc() = default;

	void solve();

private:
	Options* options;
	McParticles* particles;
};