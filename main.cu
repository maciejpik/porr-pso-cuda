#include "include/Options.cuh"
#include "include/PsoParticles.cuh"
#include "include/Pso.cuh"
#include "include/McParticles.cuh"
#include "include/Mc.cuh"

#include <cuda_runtime.h>

__constant__ int d_particlesNumber;
__constant__ int d_dimensions;
__constant__ boxConstraints d_initializationBoxConstraints;
__constant__ boxConstraints d_boxConstraints;
__constant__ psoConstants d_psoConstants;
__constant__ mcConstants d_mcConstants;

int main(int argc, char* argv[])
{
	Options* options = new Options(argc, argv);
	options->verbose = false;

	PsoParticles* particles = new PsoParticles(options);
	Pso* pso = new Pso(options, particles);
	pso->solve();

	//McParticles* particles = new McParticles(options);
	//Mc* mc = new Mc(options, particles);
	//mc->solve();
}