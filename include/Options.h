#pragma once

#include <cuda_runtime.h>

struct boxConstraints
{
	float min;
	float max;
};

class Options
{
public:
	Options();
	virtual ~Options() = default;

	int particlesNumber;
	int dimesions;
	boxConstraints initializationBoxConstraints;
	boxConstraints boxConstraints;

	void setBlockSizeInitialization(const int blockSize);
	int getBlockSizeInitialization();
	int getGridSizeInitialization();

private:
	int blockSize_initialization;
	int gridSize_initialization;
};