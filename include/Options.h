#pragma once

#include <cuda_runtime.h>

struct boxConstraints
{
	float min;
	float max;
};

struct psoConstants
{
	float w;
	float speedLocal;
	float speedGlobal;
};

class Options
{
public:
	Options::Options(int argc, char* argv[]);
	virtual ~Options() = default;

	int particlesNumber;
	int dimesions;
	boxConstraints initializationBoxConstraints;
	boxConstraints boxConstraints;
	psoConstants psoConstants;

	enum taskType {
		TASK_1,
		TASK_2
	};
	taskType task;

	void setBlockSizeInitialization(const int blockSize);
	int getBlockSizeInitialization();
	int getGridSizeInitialization();

private:
	int blockSize_initialization;
	int gridSize_initialization;
};