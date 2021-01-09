#include "../include/InputParser.h"
#include "../include/Options.h"

#include <stdio.h>

void InputParser::parse(Options* options, int argc, char* argv[])
{
	if (argc == 3)
	{
		sscanf(argv[1], "%d", &options->particlesNumber);
		sscanf(argv[2], "%d", &options->dimesions);
	}

	return;
}