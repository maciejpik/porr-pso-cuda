#pragma once

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

struct mcConstants
{
	float sigma;
	float T;
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
	mcConstants mcConstants;
	enum taskType {
		TASK_1,
		TASK_2
	};
	taskType task;
	float stopCriterion;
	bool verbose;
};