#include "include/Options.cuh"
#include "include/PsoParticles.cuh"
#include "include/Pso.cuh"
#include "include/McParticles.cuh"
#include "include/Mc.cuh"

#include <cuda_runtime.h>
#include <chrono>

__constant__ int d_particlesNumber;
__constant__ int d_dimensions;
__constant__ boxConstraints d_initializationBoxConstraints;
__constant__ boxConstraints d_solutionBoxConstraints;
__constant__ psoConstants d_psoConstants;
__constant__ mcConstants d_mcConstants;

int main(int argc, char* argv[])
{
	Options* options = new Options(argc, argv);
	options->verbose = true;

	auto tStart = std::chrono::high_resolution_clock::now();

	//PsoParticles* particles = new PsoParticles(options);
	McParticles* particles = new McParticles(options);

	auto tEnd = std::chrono::high_resolution_clock::now();
	long long int duration = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();
	printf("Initialization took %lf s\n",duration / 1000000.0);

	//Pso* pso = new Pso(options, particles);
	//pso->solve();
	//delete pso;

	Mc* mc = new Mc(options, particles);
	mc->solve();
	delete mc;

	delete particles;
	delete options;
}